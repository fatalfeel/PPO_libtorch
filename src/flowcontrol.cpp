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
	m_ppo 		= new CPPO();
}

FlowControl::~FlowControl()
{
	ServerMessage::Free();
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
