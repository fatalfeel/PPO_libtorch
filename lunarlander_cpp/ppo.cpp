#include <tuple>
#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"
#include "actorcritic.h"
#include "categorical.h"
#include "ppo.h"

CPPO::CPPO(int64_t train_epochs, double lr, std::tuple<double,double> betas, double gamma, double eps_clip)
{
	optim::AdamOptions options;
	options.lr(lr);
	options.betas(betas);

	m_train_epochs	= train_epochs;
	m_gamma 		= gamma;
	m_eps_clip		= eps_clip;

	m_policy_ac = new ActorCritic(torch::kFloat64);
	m_optimizer = new optim::Adam(m_policy_ac->parameters(), options);
}

CPPO::~CPPO()
{
	delete m_optimizer;
	delete m_policy_ac;
}

void CPPO::Train_Update(GameContent* gamedata)
{
	std::vector<Tensor>::iterator			it_reward;
	std::vector<unsigned char>::iterator	it_bterminal;
	std::vector<Tensor> 					rewards;
	Tensor 									discounted_reward;

	//debug use
	/*torch::Tensor	rdp;
	double*			data_out02;*/

	rewards				= {};
	discounted_reward	= torch::tensor(ArrayRef<double>({0.0}), torch::kFloat64);

	for(it_reward=gamedata->m_rewards.end()-1, it_bterminal=gamedata->m_bterminals.end()-1;
		it_reward!=gamedata->m_rewards.begin()-1 && it_bterminal!=gamedata->m_bterminals.begin()-1;
		it_reward--,it_bterminal--)
	{
		//debug use
		//double* data_out00 	= (double*)(*it_reward).data_ptr();
		//double* data_out01 	= (double*)discounted_reward.data_ptr();
		/*if(rewards.size()>0)
		{
			rdp			= rewards[k-1];
			data_out02 	= (double*)rdp.data_ptr();
		}*/

		if(*it_bterminal)
			discounted_reward = torch::tensor(ArrayRef<double>({0.0}), torch::kFloat64);

        discounted_reward = *it_reward + m_gamma * discounted_reward;
        rewards.insert(rewards.begin(),discounted_reward);

        /*rdp			= rewards[0];
        data_out02 	= (double*)rdp.data_ptr();*/
	}

	//debug use
	/*rdp			= rewards[3];
	data_out02 	= (double*)rdp.data_ptr();

	rdp			= rewards[2];
	data_out02 	= (double*)rdp.data_ptr();

	rdp			= rewards[1];
	data_out02 	= (double*)rdp.data_ptr();

	rdp			= rewards[0];
	data_out02 	= (double*)rdp.data_ptr();*/

	torch::Tensor curr_states      	= torch::stack(gamedata->m_states).detach();
	torch::Tensor curr_actions      = torch::stack(gamedata->m_actions).detach();
	torch::Tensor curr_actlogprobs	= torch::stack(gamedata->m_actorlogprobs).detach();
	torch::Tensor vec_rewards		= torch::stack(rewards).detach();

	torch::Tensor critic_vpi  = m_policy_ac->Critic_Forward(curr_states);
	critic_vpi  = torch::squeeze(critic_vpi);
	torch::Tensor qsa_sub_vs  = vec_rewards - critic_vpi.detach();   // A(s,a) => Q(s,a) - V(s), V(s) is critic
	torch::Tensor advantages  = (qsa_sub_vs - qsa_sub_vs.mean()) / (qsa_sub_vs.std() + 1e-5);

	int64_t		i;
	CRITICRET	cret;
	Tensor 		ratios;
	Tensor		surr1;
	Tensor 		surr2;
	Tensor 		mseloss; //mse loss
	Tensor 		ppoloss; //ppo loss (negative for reward)
	for(i=0; i<m_train_epochs; i++)
	{
		cret 	= m_policy_ac->Calculation(curr_states, curr_actions);
		ratios  = torch::exp(cret.critic_actlogprobs - curr_actlogprobs.detach());

		surr1   = ratios * advantages;
		surr2   = torch::clamp(ratios, 1.0f-m_eps_clip, 1.0f+m_eps_clip) * advantages;

		mseloss = 0.5*torch::mse_loss(cret.next_critic_values, vec_rewards, Reduction::Mean);
		ppoloss	= -torch::min(surr1, surr2) + mseloss - 0.01*cret.entropys;

		m_optimizer->zero_grad();
		ppoloss.mean().backward();

		//debug use
		/*torch::OrderedDict<std::string, torch::Tensor>param_L0 = m_policy_ac->m_L0->named_parameters();
		for (auto &p : param_L0)
		{
			if (p.key().find("weight") != std::string::npos)
			{
				double* 	data_p0 = (double*)p.value().data_ptr();
				IntArrayRef size_p0 = p.value().sizes();
				int cc = 0;
			}
		}*/

		m_optimizer->step();

		//debug use
		/*param_L0 = m_policy_ac->m_L0->named_parameters();
		for (auto &p : param_L0)
		{
			if (p.key().find("weight") != std::string::npos)
			{
				double* 	data_p0 = (double*)p.value().data_ptr();
				IntArrayRef size_p0 = p.value().sizes();
				int cc = 0;
			}
		}*/
	}
}

