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

	//debug use
	//double* data_out02 	= (double*)local_actor.data_ptr();

	return local_actor;
}

torch::Tensor ActorCritic::Critic_Forward(torch::Tensor& input)
{
	torch::Tensor local_critic = m_critic->forward(input);
	return local_critic;
}

torch::Tensor ActorCritic::Interact(torch::Tensor envstate, GameContent* gamedata)
{
	torch::Tensor 		e2d_state = envstate.reshape({1,-1}).to(torch::kFloat64);
	torch::Tensor 		act_mu 		= Actor_Forward(e2d_state);
	torch::Tensor		mat_std 	= torch::diag(m_action_std).to(torch::kFloat64);
	NormalDistribute 	distribute(act_mu, mat_std);
	torch::Tensor		action 		= distribute.Sample();
	torch::Tensor   	actlogprob;
	std::vector<Tensor> vec_act;

	for(int64_t i=0; i<m_shape; i++)
		vec_act.push_back((action[i][i]));
	action = torch::stack(vec_act).reshape({1,-1});

	actlogprob = distribute.Log_Prob(action);
	//debug use
	/*IntArrayRef s01 		= actlogprob.sizes();
	double*		data_out01 	= (double*)actlogprob.data_ptr();
	for(int i=0; i<4; i++)
	{
		std::cout << data_out01[i*4+0];
		std::cout << " ";
		std::cout << data_out01[i*4+1];
		std::cout << " ";
		std::cout << data_out01[i*4+2];
		std::cout << " ";
		std::cout << data_out01[i*4+3];
		std::cout << std::endl;
	}*/
	actlogprob = actlogprob.diagonal().sum(-1,true);

	/*double* 	data_out00 	= (double*)actlogprob.data_ptr();
	IntArrayRef s00 		= actlogprob.sizes();*/

	gamedata->m_states.push_back(envstate);
	gamedata->m_actions.push_back(action);
	gamedata->m_actorlogprobs.push_back(actlogprob);

	return action;
}

CRITICRET ActorCritic::Calculation(torch::Tensor& states, torch::Tensor& actions)
{
	CRITICRET cret;

	torch::Tensor 		entropy;
	torch::Tensor		onestate;
	torch::Tensor 		e2d_state;
	torch::Tensor		act_mu;
	torch::Tensor		mat_std;
	torch::Tensor 		critic_actlogprob;
	std::vector<Tensor> vec_actlogprobs;
	std::vector<Tensor> vec_entropys;

	//double* 	data_out02 	= (double*)states.data_ptr();
	//IntArrayRef s02			= states.sizes();

	int64_t states_size = states.size(0);
	for(int64_t i=0; i<states_size; i++)
	{
		onestate	= states[i];
		e2d_state	= onestate.reshape({1,-1});
		act_mu 		= Actor_Forward(e2d_state);
		mat_std 	= torch::diag(m_action_std).to(torch::kFloat64);

		NormalDistribute distribute(act_mu, mat_std);

		critic_actlogprob	= distribute.Log_Prob(actions[i]);
		/*IntArrayRef s01 		= critic_actlogprob.sizes();
		double*		data_out01 	= (double*)critic_actlogprob.data_ptr();
		for(int i=0; i<4; i++)
		{
			std::cout << data_out01[i*4+0];
			std::cout << " ";
			std::cout << data_out01[i*4+1];
			std::cout << " ";
			std::cout << data_out01[i*4+2];
			std::cout << " ";
			std::cout << data_out01[i*4+3];
			std::cout << std::endl;
		}*/
		critic_actlogprob 	= critic_actlogprob.diagonal().sum(-1, true);
		vec_actlogprobs.push_back(critic_actlogprob);

		entropy       		= distribute.Entropy();
		/*IntArrayRef s02 		= entropy.sizes();
		double*		data_out02 	= (double*)entropy.data_ptr();
		for(int i=0; i<4; i++)
		{
			std::cout << data_out01[i*4+0];
			std::cout << " ";
			std::cout << data_out01[i*4+1];
			std::cout << " ";
			std::cout << data_out01[i*4+2];
			std::cout << " ";
			std::cout << data_out01[i*4+3];
			std::cout << std::endl;
		}*/
		entropy 			= entropy.diagonal().sum(-1,true);
		vec_entropys.push_back(entropy);
	}

	cret.critic_actlogprobs = torch::squeeze(torch::stack(vec_actlogprobs));
	cret.entropys 			= torch::squeeze(torch::stack(vec_entropys));

	cret.next_critic_values = Critic_Forward(states);
	cret.next_critic_values	= torch::squeeze(cret.next_critic_values);

	/*double* 	data_out00 	= (double*)cret.critic_actlogprobs.data_ptr();
	IntArrayRef s00			= cret.critic_actlogprobs.sizes();

	double* 	data_out01 	= (double*)cret.entropys.data_ptr();
	IntArrayRef s01			= cret.entropys.sizes();*/

	return cret;
}

void ActorCritic::Predict_Reward(torch::Tensor& next_state, GameContent* gamedata)
{
	torch::Tensor 		e2d_state = next_state.reshape({1,-1}).to(torch::kFloat64);
	torch::Tensor 		act_mu 		= Actor_Forward(e2d_state);
	torch::Tensor		mat_std 	= torch::diag(m_action_std).to(torch::kFloat64);
	NormalDistribute 	distribute(act_mu, mat_std);
	torch::Tensor		action 		= distribute.Sample();
	torch::Tensor   	actlogprob;
	std::vector<Tensor> 					vec_act;
	std::vector<unsigned char>::iterator	vit;

	for(int64_t i=0; i<m_shape; i++)
		vec_act.push_back((action[i][i]));
	action = torch::stack(vec_act).reshape({1,-1});

	actlogprob = distribute.Log_Prob(action);
	actlogprob = actlogprob.diagonal().sum(-1,true);

	double* 	data_out00 	= (double*)actlogprob.data_ptr();
	IntArrayRef s00 		= actlogprob.sizes();

	//next_state for 1d
	torch::Tensor	next_value  = Critic_Forward(next_state);
	torch::Tensor	data_value  = next_value.detach();

	//debug use
	//double* 	data_out00 	= (double*)data_value.data_ptr();
	//IntArrayRef s00 		= data_value.sizes();
    gamedata->m_states.push_back(next_state);
    gamedata->m_actions.push_back(action);
    gamedata->m_actorlogprobs.push_back(actlogprob);
    gamedata->m_rewards.push_back(data_value);

    vit 	= gamedata->m_bterminals.end()-1;
    *vit	= true;
    gamedata->m_bterminals.push_back(true);
}


