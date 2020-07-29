#ifndef FLOWCONTROL_H
#define FLOWCONTROL_H

class FlowControl
{
public:
	FlowControl();
    ~FlowControl();

    void Start();

private:
    int 	m_started;
    CPPO*	m_ppo;
};

#endif
