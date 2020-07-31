#ifndef GAMECONTENT_H
#define GAMECONTENT_H

class GameContent
{
public:
	GameContent();
    ~GameContent();
    void ReleaseContent();

    std::vector<torch::Tensor> 	m_actions;
    std::vector<torch::Tensor> 	m_states;
    std::vector<torch::Tensor> 	m_rewards;
    std::vector<torch::Tensor> 	m_actorlogprobs;
    std::vector<unsigned char>	m_bterminals;
};

#endif
