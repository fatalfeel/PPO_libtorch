#ifndef CPPO_H
#define CPPO_H

class CPPO
{
public:
	CPPO(double lr, std::tuple<double,double> betas, double gamma, double eps_clip);
    ~CPPO();

    void Predict_Reward(torch::Tensor& next_state, GameContent& gamedata);
    void Train_Update(GameContent& gamedata);

	ActorCritic*  				m_policy_ac;
    optim::Adam* 				m_optimizer;

private:
    int							m_train_epochs;
	double 						m_gamma;
    double 						m_eps_clip;
    //torch::Device 			m_device;
};

#endif
