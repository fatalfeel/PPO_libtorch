#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"
#include "actorcritic.h"
#include "categorical.h"

ActorCritic::ActorCritic()
{
	m_actor = nn::Sequential(nn::Linear(8, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 64),
	                        nn::Functional(torch::tanh),
							nn::Linear(64, 4));

	m_critic = nn::Sequential(nn::Linear(8, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 1));

	nn::Module::register_module("ActorCritic::m_actor",  m_actor);
	nn::Module::register_module("ActorCritic::m_critic", m_critic);
}

ActorCritic::~ActorCritic()
{
	nn::Module::unregister_module("ActorCritic::m_actor");
	nn::Module::unregister_module("ActorCritic::m_critic");
}

torch::Tensor ActorCritic::Actor_Forward(torch::Tensor& input)
{
	torch::Tensor local_actor = m_actor->forward(input);
	local_actor = torch::softmax(local_actor,-1);
	return local_actor;
}

torch::Tensor ActorCritic::Critic_Forward(torch::Tensor& input)
{
	torch::Tensor local_critic = m_critic->forward(input);
	return local_critic;
}

torch::Tensor ActorCritic::Interact(torch::Tensor envstate, GameContent& gamedata)
{
	torch::Tensor 	actor_actprob 	= Actor_Forward(envstate);
	Categorical	distribute(actor_actprob);
	torch::Tensor	action 			= distribute.Sample({1});
	torch::Tensor   actlogprob  	= distribute.Log_Prob(action);

	gamedata.m_states.push_back(envstate);
	gamedata.m_actions.push_back(action);
	gamedata.m_actorlogprobs.push_back(actlogprob);

	return action.detach();
}

CRITICRET ActorCritic::Calculation(torch::Tensor& states, torch::Tensor& actions)
{
	CRITICRET cret;

	torch::Tensor critic_actprobs   = Actor_Forward(states);
	Categorical	distribute(critic_actprobs);
	cret.critic_actlogprobs			= distribute.Log_Prob(actions);
	cret.entropy            		= distribute.Entropy();
	cret.next_critic_values 		= Critic_Forward(states);
	cret.next_critic_values			= torch::squeeze(cret.next_critic_values);

	return cret;
}


