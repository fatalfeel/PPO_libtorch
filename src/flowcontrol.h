#ifndef FLOWCONTROL_H
#define FLOWCONTROL_H

class FlowControl
{
public:
	FlowControl();
    ~FlowControl();

    void Start();
    void TrainingLoop();

private:
    int 			m_started;
    CPPO*			m_ppo;
    GameContent*	m_gamedata;
};

#endif
