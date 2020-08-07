#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"
#include "actorcritic.h"
#include "normaldistribute.h"

ActorCritic::ActorCritic(torch::Dtype dtype, int64_t shape, double scale)
{
	m_actor	= nn::Sequential(nn::Linear(24, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 32),
							nn::Functional(torch::tanh),
							nn::Linear(32, 4),
							nn::Functional(torch::tanh));

	m_critic= nn::Sequential(nn::Linear(24, 64),
							nn::Functional(torch::tanh),
							nn::Linear(64, 32),
							nn::Functional(torch::tanh),
							nn::Linear(32, 1));
	//debug use
	/*m_L0 = nn::Linear(64, 32);
	m_actor = nn::Sequential(nn::Linear(24, 64),
							nn::Functional(torch::tanh),
							m_L0,
							nn::Functional(torch::tanh),
							nn::Linear(32, 4),
							nn::Functional(torch::tanh));

	m_L1 = nn::Linear(64, 32);
	m_critic = nn::Sequential(nn::Linear(24, 64),
							nn::Functional(torch::tanh),
							m_L1,
							nn::Functional(torch::tanh),
							nn::Linear(32, 1));*/

	m_actor->to(dtype);
	m_critic->to(dtype);

	nn::Module::register_module("ActorCritic::m_actor",  m_actor);
	nn::Module::register_module("ActorCritic::m_critic", m_critic);

	m_shape			= shape;
	m_action_std	= torch::full({m_shape}, scale).to(torch::kFloat64); //1x4
}

ActorCritic::~ActorCritic()
{
	nn::Module::unregister_module("ActorCritic::m_actor");
	nn::Module::unregister_module("ActorCritic::m_critic");
}

torch::Tensor ActorCritic::Actor_Forward(torch::Tensor& input)
{
	torch::Tensor local_actor = m_actor->forward(input);

	return local_actor;
}

torch::Tensor ActorCritic::Critic_Forward(torch::Tensor& input)
{
	torch::Tensor local_critic = m_critic->forward(input);

	return local_critic;
}

torch::Tensor ActorCritic::Interact(torch::Tensor envstate, GameContent* gamedata)
{
	torch::Tensor 		e2d_state 	= envstate.reshape({1,-1});
	torch::Tensor		act_mu 		= Actor_Forward(e2d_state);
	NormalDistribute distribute(act_mu, m_action_std);
	torch::Tensor		action 		= distribute.Sample();
	torch::Tensor   	actlogprob;

	actlogprob	= distribute.Log_Prob(action);
	actlogprob 	= actlogprob.sum(-1);

	gamedata->m_states.push_back(e2d_state);
	gamedata->m_actions.push_back(action);
	gamedata->m_actorlogprobs.push_back(actlogprob);

	return action.detach();
}

CRITICRET ActorCritic::Calculation(torch::Tensor& states, torch::Tensor& actions)
{
	CRITICRET cret;

	torch::Tensor		acts_mu;
	torch::Tensor 		entropy;
	torch::Tensor 		critic_actlogprob;
	torch::Tensor 		curr_states;
	std::vector<Tensor> vec_actlogprobs;
	std::vector<Tensor> vec_entropys;

	acts_mu = Actor_Forward(states);

	//align acts_mu actions size
	for(int64_t i=0; i<acts_mu.size(0) && i<actions.size(0); i++)
	{
		//Tensor mu 	= torch::tensor(ArrayRef<double>({5.1,  0.1,  9.1,  0.1}), torch::kFloat64);
		//Tensor act 	= torch::tensor(ArrayRef<double>({-0.2,  0.1, -0.2,  0.1}), torch::kFloat64);
		NormalDistribute distribute(acts_mu[i], m_action_std);

		critic_actlogprob	= distribute.Log_Prob(actions[i]);
		critic_actlogprob 	= critic_actlogprob.sum(-1);
		vec_actlogprobs.push_back(critic_actlogprob);

		entropy       		= distribute.Entropy();
		entropy 			= entropy.sum(-1);
		vec_entropys.push_back(entropy);
	}

	cret.critic_actlogprobs = torch::stack(vec_actlogprobs);
	cret.entropys 			= torch::stack(vec_entropys);

	/*std::cout << "critic_actlogprobs::\n" << cret.critic_actlogprobs << std::endl;
	std::cout << "entropys::\n" << cret.entropys << std::endl;
	std::cout << std::endl;*/

	cret.next_critic_values = Critic_Forward(states);
	cret.next_critic_values	= torch::squeeze(cret.next_critic_values);

	return cret;
}

void ActorCritic::Predict_Reward(torch::Tensor& next_state, GameContent* gamedata, double gamma)
{
	//torch::Tensor 		e2d_state 	= next_state.reshape({1,-1}).to(torch::kFloat64);
	//torch::Tensor 		act_mu 		= Actor_Forward(e2d_state);
	//NormalDistribute distribute(act_mu, m_action_std);
	//torch::Tensor		action 		= distribute.Sample();
	//torch::Tensor   	actlogprob;
	torch::Tensor	next_value  = Critic_Forward(next_state);
	torch::Tensor	data_value  = next_value.detach();
	std::vector<unsigned char>::iterator vit_bterminal;
	std::vector<torch::Tensor>::iterator vit_reward;

	vit_bterminal = gamedata->m_bterminals.end()-1;
	if( *vit_bterminal == false )
	{
		vit_reward = gamedata->m_rewards.end()-1;
		*vit_reward = *vit_reward + gamma*data_value;
	}
}
