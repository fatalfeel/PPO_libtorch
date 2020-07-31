#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H

typedef struct _CRITICRET
{
	torch::Tensor critic_actlogprobs;
	torch::Tensor next_critic_values;
	torch::Tensor entropy;
}CRITICRET;

class ActorCritic : public nn::Module
{
public:
    ActorCritic(torch::Dtype dtype);
    ~ActorCritic();

    torch::Tensor 	Actor_Forward(torch::Tensor& input);
    torch::Tensor	Critic_Forward(torch::Tensor& input);
    int64_t			Interact(torch::Tensor envstate, GameContent* gamedata);
    CRITICRET		Calculation(torch::Tensor& states, torch::Tensor& actions);
    void 			Predict_Reward(torch::Tensor& next_state, GameContent* gamedata);

    nn::Sequential	m_actor;
    nn::Sequential 	m_critic;
    //nn::Linear      m_L0{nullptr}; //debug use
    //nn::Linear		m_L1{nullptr}; //debug use
};

#endif
