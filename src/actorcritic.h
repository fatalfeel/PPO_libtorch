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
    ActorCritic();
    ~ActorCritic();

    torch::Tensor 	Actor_Forward(torch::Tensor& input);
    torch::Tensor	Critic_Forward(torch::Tensor& input);
    torch::Tensor	Interact(torch::Tensor envstate, GameContent& gamedata);
    CRITICRET		Calculation(torch::Tensor& states, torch::Tensor& actions);

    nn::Sequential	m_actor;
    nn::Sequential 	m_critic;
};

#endif
