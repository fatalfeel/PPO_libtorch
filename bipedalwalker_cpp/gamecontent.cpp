#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"

GameContent::GameContent()
{

}

GameContent::~GameContent()
{
	ReleaseContent();
}

void GameContent::ReleaseContent()
{
	m_actions.clear();
	m_states.clear();
	m_rewards.clear();
	m_actorlogprobs.clear();
	m_bterminals.clear();
}

