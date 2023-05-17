import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC

class PDA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		
	def tangent_vector(self, obs):
		# input: obs
		# output: random_shift_1pixel(obs) - obs
		
		pad = nn.Sequential(torch.nn.ReplicationPad2d(1))
		pad_obs = pad(obs)
		
		index = np.random.randint(4, size=1)[0]
		top_left_dicts = {0: [1,2], 1: [1,0], 2: [2,1], 3:[0,1]}
		top, left = top_left_dicts[index]		

		obs_aug = torchvision.transforms.functional.crop(pad_obs, top=top, left=left, height=obs.shape[-1], width=obs.shape[-1])

		tan_vector = obs_aug - obs
        
		return tan_vector

	def tangent_vector_overlay(self, original_obs, imgs):
		# input: original_obs, images from place365 dataset
		# output: original_obs - imgs

		# original_obs: [0, 255]
		# imgs: [0, 1]
		tan_vector_overlay = (original_obs/255. - imgs) * 255.
		return tan_vector_overlay

	def update_actor_and_alpha(self, original_obs, obs_aug1, obs_aug2, imgs, obs_DA, L=None, step=None, update_alpha=True):
		# actor loss for obs_aug1
		mu_aug1, pi_aug1, log_pi_aug1, log_std_aug1 = self.actor(obs_aug1, detach=True)
		std_aug1 = log_std_aug1.exp()
		actor_Q1_aug1, actor_Q2_aug1 = self.critic(obs_aug1, pi_aug1, detach=True)
		actor_Q_aug1 = torch.min(actor_Q1_aug1, actor_Q2_aug1)
		actor_loss_shift = (self.alpha.detach() * log_pi_aug1 - actor_Q_aug1).mean()
		actor_loss = actor_loss_shift

		if self.args.actor_DA == True:
			# actor loss for obs_DA
			mu_DA, pi_DA, log_pi_DA, log_std_DA = self.actor(obs_DA, detach=True)
			std_DA = log_std_DA.exp()
			actor_Q1_DA, actor_Q2_DA = self.critic(obs_DA, pi_DA, detach=True)
			actor_Q_DA = torch.min(actor_Q1_DA, actor_Q2_DA)
			actor_loss_DA = (self.alpha.detach() * log_pi_DA - actor_Q_DA).mean()

			actor_loss += actor_loss_DA
		else:
			obs_aug2_DA = (0.5*(obs_aug2/255.) + 0.5*imgs)*255.
			mu_DA, _, _, log_std_DA = self.actor(obs_aug2_DA, detach=True)
			std_DA = log_std_DA.exp()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
		
		# KL loss: KL(A(obs_aug1), A(obs_aug2))
		# mu and mu_aug has been squashed (tanh), no further 'torch.tanh' is required.
		mu_aug2, _, _, log_std_aug2 = self.actor(obs_aug2, compute_pi=False, compute_log_pi=False, detach=True)
		std_aug2 = log_std_aug2.exp()
		# detach first
		dist_aug1 = torch.distributions.Normal(mu_aug1.detach(), std_aug1.detach())
		dist_aug2 = torch.distributions.Normal(mu_aug2, std_aug2)		
		if self.args.actor_KL_aug in ['random_shift', 'complex_DA']:
			KL_loss_shift = torch.mean(torch.distributions.kl_divergence(dist_aug1, dist_aug2))
		else:
			with torch.no_grad():
				KL_loss_shift = torch.mean(torch.distributions.kl_divergence(dist_aug1, dist_aug2))	

		if self.args.actor_KL_aug in ['random_overlay', 'complex_DA']:
			# KL loss for obs_DA
			dist_DA = torch.distributions.Normal(mu_DA, std_DA)
			KL_loss_DA = torch.mean(torch.distributions.kl_divergence(dist_aug1, dist_DA))
		else:
			with torch.no_grad():
				dist_DA = torch.distributions.Normal(mu_DA, std_DA)
				KL_loss_DA = torch.mean(torch.distributions.kl_divergence(dist_aug1, dist_DA))
		
		if self.args.actor_KL_aug == 'random_shift': KL_loss = KL_loss_shift
		if self.args.actor_KL_aug == 'random_overlay': KL_loss = KL_loss_DA
		if self.args.actor_KL_aug == 'complex_DA': KL_loss = (KL_loss_shift + KL_loss_DA) / 2

		actor_loss += self.args.actor_KL_weight * KL_loss
		
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha * (-log_pi_aug1 - self.target_entropy).detach()).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()

		return actor_loss.detach().cpu().numpy(), alpha_loss.detach().cpu().numpy(), self.alpha.detach().cpu().numpy(), KL_loss_shift.detach().cpu().numpy(), KL_loss_DA.detach().cpu().numpy()

	def update_critic(self, original_obs, obs_aug1, obs_aug2, imgs, obs_DA, action, reward, original_next_obs, next_obs_aug1, next_obs_aug2, not_done, L=None, step=None):
		with torch.no_grad():
			if self.args.svea_target:
				# target Q for (original_next_obs, action)
				_, policy_action, log_pi, _ = self.actor(original_next_obs)
				target_Q1, target_Q2 = self.critic_target(original_next_obs, policy_action)
				target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
				target_Q = reward + (not_done * self.discount * target_V)
			else:
				# target Q for (next_obs_aug1, action)
				_, policy_action_aug1, log_pi_aug1, _ = self.actor(next_obs_aug1)
				target_Q1_aug1, target_Q2_aug1 = self.critic_target(next_obs_aug1, policy_action_aug1)
				target_V_aug1 = torch.min(target_Q1_aug1, target_Q2_aug1) - self.alpha.detach() * log_pi_aug1
				target_Q_aug1 = reward + (not_done * self.discount * target_V_aug1)

				# target Q for (next_obs_aug2, action)
				_, policy_action_aug2, log_pi_aug2, _ = self.actor(next_obs_aug2)
				target_Q1_aug2, target_Q2_aug2 = self.critic_target(next_obs_aug2, policy_action_aug2)
				target_V_aug2 = torch.min(target_Q1_aug2, target_Q2_aug2) - self.alpha.detach() * log_pi_aug2
				target_Q_aug2 = reward + (not_done * self.discount * target_V_aug2)

				# avg target Q
				target_Q = (target_Q_aug1 + target_Q_aug2) / 2

		# ||Q(random_shift(obs)) - target||^2
		if self.args.tan_prop_weight > 0:
			obs_aug1.requires_grad = True
		current_Q1_aug1, current_Q2_aug1 = self.critic(obs_aug1, action)
		critic_loss = self.svea_alpha * (F.mse_loss(current_Q1_aug1, target_Q) + F.mse_loss(current_Q2_aug1, target_Q))
		if not self.args.svea_target:
			current_Q1_aug2, current_Q2_aug2 = self.critic(obs_aug2, action)
			critic_loss += self.svea_alpha * (F.mse_loss(current_Q1_aug2, target_Q) + F.mse_loss(current_Q2_aug2, target_Q))
			critic_loss = critic_loss / 2

		if self.args.tan_prop_weight > 0:
			# calculate the Jacobian matrix for non-linear model
			Q = torch.min(current_Q1_aug1, current_Q2_aug1)
			jacobian = torch.autograd.grad(outputs=Q, inputs=obs_aug1, grad_outputs=torch.ones(Q.size(), device=self.device), retain_graph=True, create_graph=True)[0]

			# calculate tangent vector
			with torch.no_grad():
				# calculate the tangent vector
				tangent_vector = self.tangent_vector(obs_aug1)

			# tangent_loss = ||sum over all pixels(jacobian * (obs_shift1 - obs)) - 0||^2
			tangent_loss = torch.mean(torch.square(torch.sum((jacobian * tangent_vector),(3,2,1))), dim=-1)
			tangent_loss_numpy = tangent_loss.detach().cpu().numpy()
			
			# add tangent prop regularization in critic
			critic_loss += self.args.tan_prop_weight * tangent_loss
			obs_aug1.requires_grad = False
		
			if L is not None:
				L.log('train_critic/tangent_prop_loss', tangent_loss, step)	
		else:
			tangent_loss_numpy = 0		
		
		# ||Q(complex_DA(obs)) - target||^2
		current_Q1_DA, current_Q2_DA = self.critic(obs_DA, action)
		critic_loss += self.svea_beta * (F.mse_loss(current_Q1_DA, target_Q) + F.mse_loss(current_Q2_DA, target_Q))

		if self.args.tan_prop_overlay_weight > 0:
			# calculate the Jacobian matrix of obs_overlay for non-linear model
			rand_alpha = np.random.uniform(self.args.alpha_min, self.args.alpha_max)
			obs_DA_rand = ((1-rand_alpha)*(obs_aug1/255.) + (rand_alpha)*imgs)*255.
			obs_DA_rand.requires_grad = True
			current_Q1_DA_rand, current_Q2_DA_rand = self.critic(obs_DA_rand, action)
			
			Q_overlay = torch.min(current_Q1_DA_rand, current_Q2_DA_rand)
			jacobian_overlay = torch.autograd.grad(outputs=Q_overlay, inputs=obs_DA_rand, grad_outputs=torch.ones(Q_overlay.size(), device=self.device), retain_graph=True, create_graph=True)[0]

			# calculate tangent vector of obs_overlay
			with torch.no_grad():
				# calculate the tangent vector
				tangent_vector_overlay = self.tangent_vector_overlay(obs_aug1, imgs)

			# tangent_loss_overlay = ||sum over all pixels(jacobian_overlay * tangent_vector_overlay) - 0||^2
			tangent_loss_overlay = torch.mean(torch.square(torch.sum((jacobian_overlay * tangent_vector_overlay), (3,2,1))), dim=-1)
			tangent_loss_overlay_numpy = tangent_loss_overlay.detach().cpu().numpy()
			
			# add tangent prop (overlay) regularization in critic
			critic_loss += self.args.tan_prop_overlay_weight * tangent_loss_overlay
			obs_DA_rand.requires_grad = False
		else:
			tangent_loss_overlay_numpy = 0
		
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		return critic_loss.detach().cpu().numpy(), tangent_loss_numpy, tangent_loss_overlay_numpy

	def update(self, replay_buffer, L, step):
		original_obs, obs_aug1, obs_aug2, imgs, obs_DA, action, reward, original_next_obs, next_obs_aug1, next_obs_aug2, not_done = replay_buffer.sample_pda(da=self.args.complex_DA)

		critic_loss, tangent_loss, tangent_loss_overlay = self.update_critic(original_obs, obs_aug1, obs_aug2, imgs, obs_DA, action, reward, original_next_obs, next_obs_aug1, next_obs_aug2, not_done, L, step)

		actor_loss, alpha_loss, alpha_value, KL_loss_shift, KL_loss_DA = None, None, None, None, None
		if step % self.actor_update_freq == 0:
			actor_loss, alpha_loss, alpha_value, KL_loss_shift, KL_loss_DA = self.update_actor_and_alpha(original_obs, obs_aug1, obs_aug2, imgs, obs_DA, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		return critic_loss, tangent_loss, tangent_loss_overlay, actor_loss, alpha_loss, alpha_value, KL_loss_shift, KL_loss_DA