#include <tuple>
#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"
#include "actorcritic.h"
#include "normaldistribute.h"
#include "ppo.h"

CPPO::CPPO(int64_t train_epochs, double lr, std::tuple<double,double> betas, double gamma, double eps_clip, double vloss_coef, double entropy_coef)
{
	optim::AdamOptions options;
	options.lr(lr);
	options.betas(betas);

	m_train_epochs	= train_epochs;
	m_gamma 		= gamma;
	m_eps_clip		= eps_clip;
	m_vloss_coef	= vloss_coef;
	m_entropy_coef	= entropy_coef;

	m_policy_ac = new ActorCritic(torch::kFloat64, 4, (double)0.5f);
	m_optimizer = new optim::Adam(m_policy_ac->parameters(), options);
}

CPPO::~CPPO()
{
	delete m_optimizer;
	delete m_policy_ac;
}

void CPPO::Train_Update(GameContent* gamedata)
{
	std::vector<torch::Tensor>::iterator	it_reward;
	std::vector<unsigned char>::iterator	it_bterminal;
	std::vector<torch::Tensor> 				returns;
	Tensor 									discounted_reward;

	//debug use
	/*torch::Tensor	rdp;
	double*			data_out00;
	IntArrayRef 	s00;*/

	returns				= {};
	discounted_reward	= torch::tensor(ArrayRef<double>({0.0}), torch::kFloat64);

	for(it_reward=gamedata->m_rewards.end()-1, it_bterminal=gamedata->m_bterminals.end()-1;
		it_reward!=gamedata->m_rewards.begin()-1 && it_bterminal!=gamedata->m_bterminals.begin()-1;
		it_reward--,it_bterminal--)
	{
		//debug use
		/*if(returns.size()>0)
		{
			rdp			= returns[returns.size()-1];
			data_out00 	= (double*)rdp.data_ptr();
		}*/

		if(*it_bterminal)
			discounted_reward = torch::tensor(ArrayRef<double>({0.0}), torch::kFloat64);

        discounted_reward = *it_reward + m_gamma * discounted_reward;
        returns.insert(returns.begin(),discounted_reward);

        /*rdp			= returns[0];
        data_out00 	= (double*)rdp.data_ptr();
        s00 		= rdp.sizes();*/
	}

	torch::Tensor curr_states      	= torch::squeeze(torch::stack(gamedata->m_states)).detach();
	torch::Tensor curr_actions      = torch::squeeze(torch::stack(gamedata->m_actions)).detach();
	torch::Tensor curr_actlogprobs	= torch::squeeze(torch::stack(gamedata->m_actorlogprobs)).detach();
	torch::Tensor vec_returns		= torch::squeeze(torch::stack(returns)).detach();

	torch::Tensor critic_vpi  	= m_policy_ac->Critic_Forward(curr_states);
	critic_vpi  				= torch::squeeze(critic_vpi);
	torch::Tensor qsa_sub_vs  	= vec_returns - critic_vpi.detach();   // A(s,a) => Q(s,a) - V(s), V(s) is critic
	torch::Tensor advantages  	= (qsa_sub_vs - qsa_sub_vs.mean()) / (qsa_sub_vs.std() + 1e-5);

	int64_t		i;
	Tensor 		ratios;
	Tensor		surr1;
	Tensor 		surr2;
	Tensor		value_predict_clip;
	Tensor		value_predict_loss;
	Tensor		value_critic_loss;
	Tensor 		value_loss;
	Tensor 		ppoloss; //ppo loss (negative for reward)
	CRITICRET	cret;

	for(i=0; i<m_train_epochs; i++)
	{
		cret 	= m_policy_ac->Calculation(curr_states, curr_actions);
		ratios  = torch::exp(cret.critic_actlogprobs - curr_actlogprobs.detach());

		surr1   = ratios * advantages;
		surr2   = torch::clamp(ratios, 1.0f-m_eps_clip, 1.0f+m_eps_clip) * advantages;

		//we get all samples of critic_values, so value_preds_batch equal critic_values
		/*value_predict_clip  = critic_vpi.detach() + (next_critic_values - critic_vpi.detach()).clamp(-self.eps_clip, self.eps_clip)
		  value_predict_loss  = self.MseLoss(value_predict_clip, returns)
		  value_critic_loss   = self.MseLoss(next_critic_values, returns)
		  value_loss          = 0.5 * torch.max(value_predict_loss, value_critic_loss) #combine 2 biggest loss, not pick one of them*/
		//mseloss = 0.5*torch::mse_loss(cret.next_critic_values, vec_returns, Reduction::None);
		value_predict_clip	= critic_vpi.detach() + (cret.next_critic_values - critic_vpi.detach()).clamp(-m_eps_clip, m_eps_clip);
		value_predict_loss  = torch::mse_loss(value_predict_clip, vec_returns, Reduction::None);
		value_critic_loss   = torch::mse_loss(cret.next_critic_values, vec_returns, Reduction::None);
		value_loss          = (double)0.5 * torch::max(value_predict_loss, value_critic_loss);

		ppoloss	= -torch::min(surr1, surr2) + m_vloss_coef*value_loss - m_entropy_coef*cret.entropys;

		m_optimizer->zero_grad();
		ppoloss.mean().backward();

		//debug use
		/*torch::OrderedDict<std::string, torch::Tensor>param_L0 = m_policy_ac->m_L0->named_parameters();
		for (auto &p : param_L0)
		{
			if (p.key().find("weight") != std::string::npos)
			{
				IntArrayRef size_p0 = p.value().sizes();
				std::cout << p.value() << std::endl;
			}
		}*/

		m_optimizer->step();

		//debug use
		/*param_L0 = m_policy_ac->m_L0->named_parameters();
		for (auto &p : param_L0)
		{
			if (p.key().find("weight") != std::string::npos)
			{
				IntArrayRef size_p0 = p.value().sizes();
				std::cout << p.value() << std::endl;
			}
		}*/
	}
}

