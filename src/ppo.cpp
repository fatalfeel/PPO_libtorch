#include <tuple>
#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"
#include "actorcritic.h"
#include "categorical.h"
#include "ppo.h"

CPPO::CPPO(double lr, std::tuple<double,double> betas, double gamma, double eps_clip)
{
	optim::AdamOptions options;
	options.lr(lr);
	options.betas(betas);

	m_gamma 	= gamma;
	m_eps_clip	= eps_clip;

	m_policy_ac = new ActorCritic();
	m_optimizer = new optim::Adam(m_policy_ac->parameters(), options);
}

CPPO::~CPPO()
{
	delete m_optimizer;
	delete m_policy_ac;
}

void CPPO::Predict_Reward(torch::Tensor& next_state, GameContent& gamedata)
{
	torch::Tensor 	actor_actprob 	= m_policy_ac->Actor_Forward(next_state);
	Categorical		distribute(actor_actprob);
	torch::Tensor	action 			= distribute.Sample({1});
	torch::Tensor   actlogprob  	= distribute.Log_Prob(action);
	torch::Tensor	next_value  	= m_policy_ac->Critic_Forward(next_state);
	torch::Tensor	data_value  	= next_value.detach();
	std::vector<unsigned char>::iterator vit;

    gamedata.m_states.push_back(next_state);
    gamedata.m_actions.push_back(action);
    gamedata.m_actorlogprobs.push_back(actlogprob);
    gamedata.m_rewards.push_back(data_value);

    vit 	= gamedata.m_bterminals.end()-1;
    *vit	= true;
    gamedata.m_bterminals.push_back(true);
}

void CPPO::Train_Update(GameContent& gamedata)
{
	std::vector<torch::Tensor>::iterator 	it_reward;
	std::vector<unsigned char>::iterator	it_bterminal;
	std::vector<torch::Tensor> 				rewards;
	Tensor 									discounted_reward;

	discounted_reward = torch::zeros({0});

	for(it_reward=gamedata.m_rewards.end()-1, it_bterminal=gamedata.m_bterminals.end()-1;
		it_reward!=gamedata.m_rewards.begin()-1 && it_bterminal!=gamedata.m_bterminals.begin()-1;
		it_reward--,it_bterminal--)
	{
        if(*it_bterminal)
            discounted_reward.zero_();

        discounted_reward = *it_reward + m_gamma * discounted_reward;
        rewards.insert(rewards.begin(),discounted_reward);
	}

	torch::Tensor curr_states      	= torch::stack(gamedata.m_states).detach();
	torch::Tensor curr_actions      = torch::stack(gamedata.m_actions).detach();
	torch::Tensor curr_actlogprobs	= torch::stack(gamedata.m_actorlogprobs).detach();
	torch::Tensor stack_rewards		= torch::stack(rewards).detach();

	torch::Tensor critic_vpi  = m_policy_ac->Critic_Forward(curr_states);
	critic_vpi  = torch::squeeze(critic_vpi);
	torch::Tensor qsa_sub_vs  = stack_rewards - critic_vpi.detach();   // A(s,a) => Q(s,a) - V(s), V(s) is critic
	torch::Tensor advantages  = (qsa_sub_vs - qsa_sub_vs.mean()) / (qsa_sub_vs.std() + 1e-5);

	torch::Tensor 	ratios;
	torch::Tensor	surr1;
	torch::Tensor 	surr2;
	torch::Tensor 	mloss; //mse
	torch::Tensor 	ploss; //ppo
	CRITICRET		cret;
	for(int i=0; i<m_train_epochs; i++)
	{
		cret 	= m_policy_ac->Calculation(curr_states, curr_actions);
		ratios  = torch::exp(cret.critic_actlogprobs - curr_actlogprobs.detach());

		surr1   = ratios * advantages;
		surr2   = torch::clamp(ratios, 1.0-m_eps_clip, 1.0+m_eps_clip) * advantages;

		mloss 	= 0.5*torch::mse_loss(stack_rewards, cret.next_critic_values, Reduction::Mean);
		ploss   = -torch::min(surr1, surr2) + mloss - 0.01*cret.entropy;

		m_optimizer->zero_grad();
		ploss.mean().backward();
		m_optimizer->step();
	}
}

