import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC


class SVEA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta

	def update_actor_and_alpha(self, original_obs, obs, L=None, step=None, update_alpha=True):
		mu, pi, log_pi, log_std = self.actor(obs, detach=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
			entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
												) + log_std.sum(dim=-1)

		with torch.no_grad():
			# record KL loss for \pi(obs) and \pi(obs_shift)
			obs_aug = augmentations.random_shift(original_obs)
			# mu and mu_aug has been squashed (tanh), no further 'torch.tanh' is required.
			mu_aug, _, _, log_std_aug = self.actor(obs_aug, compute_pi=False, compute_log_pi=False, detach=True)
			# std = exp(log_std)
			std = log_std.exp()
			std_aug = log_std_aug.exp()
			# detach first
			dist = torch.distributions.Normal(mu.detach(), std.detach())
			dist_aug = torch.distributions.Normal(mu_aug, std_aug)
			KL_loss_shift = torch.mean(torch.distributions.kl_divergence(dist, dist_aug))

			# record KL loss for \pi(obs) and \pi(obs_DA)
			'''
			if self.args.complex_DA == 'random_conv':
				obs_DA = augmentations.random_conv(obs_aug)
			if self.args.complex_DA == 'random_overlay':
				obs_DA = augmentations.random_overlay(obs_aug)
			'''
			obs_DA = augmentations.random_conv(obs_aug)
			mu_DA, _, _, log_std_DA = self.actor(obs_DA, compute_pi=False, compute_log_pi=False, detach=True)
			std_DA = log_std_DA.exp()
			dist_DA = torch.distributions.Normal(mu_DA, std_DA)
			KL_loss_DA = torch.mean(torch.distributions.kl_divergence(dist, dist_DA))
		
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()

		return actor_loss.detach().cpu().numpy(), alpha_loss.detach().cpu().numpy(), self.alpha.detach().cpu().numpy(), KL_loss_shift.detach().cpu().numpy(), KL_loss_DA.detach().cpu().numpy()

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			if self.args.complex_DA == 'random_conv':
				obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
			if self.args.complex_DA == 'random_overlay':
				obs = utils.cat(obs, augmentations.random_overlay(obs.clone()))
			
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			if self.args.complex_DA == 'random_conv':
				obs_aug = augmentations.random_conv(obs.clone())
			if self.args.complex_DA == 'random_overlay':
				obs_aug = augmentations.random_overlay(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		return critic_loss.detach().cpu().numpy()

	def update(self, replay_buffer, L, step):
		original_obs, obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

		critic_loss = self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		actor_loss, alpha_loss, alpha_value, KL_loss_shift, KL_loss_DA = None, None, None, None, None
		if step % self.actor_update_freq == 0:
			actor_loss, alpha_loss, alpha_value, KL_loss_shift, KL_loss_DA = self.update_actor_and_alpha(original_obs, obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		return critic_loss, actor_loss, alpha_loss, alpha_value, KL_loss_shift, KL_loss_DA