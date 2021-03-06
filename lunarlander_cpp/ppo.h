#ifndef CPPO_H
#define CPPO_H

class CPPO
{
public:
	CPPO(int64_t train_epochs, double lr, std::tuple<double,double> betas, double gamma, double eps_clip, double vloss_coef, double entropy_coef);
    ~CPPO();
    void 			Train_Update(GameContent* gamedata);

    ActorCritic*  	m_policy_ac;
    optim::Adam* 	m_optimizer;

private:
    int64_t			m_train_epochs;
	double 			m_gamma;
    double 			m_eps_clip;
    double 			m_vloss_coef;
    double 			m_entropy_coef;
    //torch::Device m_device;
};

#endif
