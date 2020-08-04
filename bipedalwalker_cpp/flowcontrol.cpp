#include <tuple>
#include <pthread.h>
#include "torch/torch.h"
using namespace torch;
#include "server_cmd.h"
#include "link_list.h"
#include "wxlocker.h"
#include "servermessage.h"
#include "gamecontent.h"
#include "actorcritic.h"
#include "ppo.h"
#include "flowcontrol.h"

FlowControl::FlowControl()
{
	m_started 	= 0;
	m_gamedata	= new GameContent();
	m_ppo 		= new CPPO((int64_t)40, (double)0.0001f, std::make_tuple((double)0.9f, (double)0.999f), (double)0.99f, (double)0.2f);
}

FlowControl::~FlowControl()
{
	ServerMessage::Free();

	delete m_ppo;
	delete m_gamedata;
}

void FlowControl::Start()
{
	if( m_started <= 0 )
	{
		m_started = 1;
		ServerMessage::Get()->InitServerSocket(SERVER_PORT);
		ServerMessage::Get()->StartServerThread();
	}
}

void FlowControl::TrainingTest()
{
	double solved_reward   	= 230;
	int log_interval    	= 20;
	int max_episodes    	= 50000;
	int max_timesteps   	= 1500;
	int avg_length      	= 0;
	int timestep        	= 0;
	double	running_reward  = 0;
	Tensor	action;

	//simulator reward
	Tensor envstate = torch::tensor(ArrayRef<double>({0.0085665,  0.00390062, 0.00911026, 0.01548401, -0.28943464,
													 -0.63517225, 1.46211898, 0.79200919, 1.0,         0.2875464,
													 -0.14784433, 0.17633283, 0.35682468, 1.0,         0.45404965,
													  0.45920607, 0.47527719, 0.50424916, 0.55013949,  0.62055016,
													  0.73044139, 0.66274732, 1.0 ,       1.0 }), torch::kFloat64);
	Tensor reward 	= torch::tensor(ArrayRef<double>({-0.17278488148785262}), torch::kFloat64);

	action = m_ppo->m_policy_ac->Interact(envstate, m_gamedata);
	m_gamedata->m_rewards.push_back(reward);
	m_gamedata->m_bterminals.push_back(false);

	action = m_ppo->m_policy_ac->Interact(envstate, m_gamedata);
	m_gamedata->m_rewards.push_back(reward);
	m_gamedata->m_bterminals.push_back(false);

	action = m_ppo->m_policy_ac->Interact(envstate, m_gamedata);
	m_gamedata->m_rewards.push_back(reward);
	m_gamedata->m_bterminals.push_back(false);

	action = m_ppo->m_policy_ac->Interact(envstate, m_gamedata);
	m_gamedata->m_rewards.push_back(reward);
	m_gamedata->m_bterminals.push_back(false);

	m_ppo->m_policy_ac->Predict_Reward(envstate, m_gamedata);
	m_ppo->Train_Update(m_gamedata);

	m_gamedata->ReleaseContent();
}

