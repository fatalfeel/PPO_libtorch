#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"
#include "actorcritic.h"
#include "categorical.h"

ActorCritic::ActorCritic(torch::Dtype dtype)
{
	m_actor	= nn::Sequential(nn::Linear(8, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 4));

	m_critic= nn::Sequential(nn::Linear(8, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 1));
	//debug use
	/*m_L0 = nn::Linear(64, 64);
	m_actor = nn::Sequential(nn::Linear(8, 64),
							nn::Functional(torch::tanh),
							m_L0,
	                        nn::Functional(torch::tanh),
							nn::Linear(64, 4));

	m_L1 = nn::Linear(64, 64);
	m_critic = nn::Sequential(nn::Linear(8, 64),
							nn::Functional(torch::tanh),
							m_L1,
							nn::Functional(torch::tanh),
							nn::Linear(64, 1));*/

	m_actor->to(dtype);
	m_critic->to(dtype);

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

	//debug use
	//double* data_out02 	= (double*)local_actor.data_ptr();

	local_actor = torch::softmax(local_actor, -1, torch::kFloat64);

	//data_out02 	= (double*)local_actor.data_ptr();

	return local_actor;
}

torch::Tensor ActorCritic::Critic_Forward(torch::Tensor& input)
{
	torch::Tensor local_critic = m_critic->forward(input);
	return local_critic;
}

int64_t ActorCritic::Interact(torch::Tensor envstate, GameContent* gamedata)
{
	torch::Tensor 	actor_actprob 	= Actor_Forward(envstate);
	Categorical	distribute(actor_actprob);
	torch::Tensor	action 			= distribute.Sample();
	torch::Tensor   actlogprob  	= distribute.Log_Prob(action);

	//debug use
	/*IntArrayRef s00 = actor_actprob.sizes();
	IntArrayRef s01 = action.sizes();
	IntArrayRef s02 = actlogprob.sizes();
	double*		data_out00 	= (double*)actor_actprob.data_ptr();
	int64_t*	data_out01 	= (int64_t*)action.data_ptr();
	double*		data_out02 	= (double*)actlogprob.data_ptr();
	int64_t		iact 		= action.detach().item().toLong();*/

	gamedata->m_states.push_back(envstate);
	gamedata->m_actions.push_back(action);
	gamedata->m_actorlogprobs.push_back(actlogprob);

	return action.detach().item().toLong();
}

CRITICRET ActorCritic::Calculation(torch::Tensor& states, torch::Tensor& actions)
{
	CRITICRET cret;

	torch::Tensor critic_actprobs   = Actor_Forward(states);
	Categorical	distribute(critic_actprobs);
	cret.critic_actlogprobs			= distribute.Log_Prob(actions);
	cret.entropys            		= distribute.Entropy();
	cret.next_critic_values 		= Critic_Forward(states);
	cret.next_critic_values			= torch::squeeze(cret.next_critic_values);

	//debug use
	/*double* data_out00 	= (double*)cret.critic_actlogprobs.data_ptr();
	IntArrayRef s00 		= cret.critic_actlogprobs.sizes();*/

	return cret;
}

void ActorCritic::Predict_Reward(torch::Tensor& next_state, GameContent* gamedata, double gamma)
{
	//torch::Tensor	actor_actprob 	= Actor_Forward(next_state);
	//Categorical	distribute(actor_actprob);
	//torch::Tensor	action 			= distribute.Sample();
	//torch::Tensor actlogprob  	= distribute.Log_Prob(action);
	torch::Tensor	next_value  	= Critic_Forward(next_state);
	torch::Tensor	data_value  	= next_value.detach();
	std::vector<unsigned char>::iterator vit_bterminal;
	std::vector<torch::Tensor>::iterator vit_reward;

    vit_bterminal = gamedata->m_bterminals.end()-1;
	if( *vit_bterminal == false )
	{
		vit_reward = gamedata->m_rewards.end()-1;
		*vit_reward = *vit_reward + gamma*data_value;
	}
}


