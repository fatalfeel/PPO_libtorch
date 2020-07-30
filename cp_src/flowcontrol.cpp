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
	m_ppo 		= new CPPO((int64_t)4, (double)0.0005f, std::make_tuple((double)0.9f, (double)0.999f), (double)0.99f, (double)0.2f);
	m_gamedata	= new GameContent();
}

FlowControl::~FlowControl()
{
	ServerMessage::Free();

	delete m_gamedata;
	delete m_ppo;
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

void FlowControl::TrainingLoop()
{
	double solved_reward   	= 230;
	int log_interval    	= 20;
	int max_episodes    	= 50000;
	int max_timesteps   	= 1500;
	int avg_length      	= 0;
	int timestep        	= 0;
	double running_reward  	= 0;
	int64_t	action			= 0;

	//simulator reward
	Tensor envstate = torch::tensor(ArrayRef<float>({0.00984659,1.3989865,0.4940299,-0.25793448,-0.01182567,-0.12153457,0.0,0.0}));
	Tensor reward 	= torch::tensor(ArrayRef<float>({1.1590785142386266}));

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
}

