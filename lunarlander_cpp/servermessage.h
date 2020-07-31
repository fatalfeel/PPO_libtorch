#ifndef		SERVER_MESSAGE_H
#define 	SERVER_MESSAGE_H

typedef void* (*THREADFUNC_PTR)(void*);

class ServerMessage
{
public:
	static ServerMessage* Get();
	static void Free();
	ServerMessage();
	~ServerMessage();
	//void			close_sigint(int dummy);
	void 			SocketProcess(int type, NODE_SOCKET* node_ptr);
	void			InitServerSocket(int port);
	int 			MagicPadCheck(unsigned long long* magic, unsigned long long onebyte);
	void* 			ServerThreadLoop();

	static void*	ThreadInterface(void* param);
	void			StartServerThread();
	void			ExternalJob(NODE_SOCKET* node_ptr);
	void			WriteData(NODE_SOCKET* node_ptr);

private:
	int					m_serversocket;
	volatile int		m_thread_work;
	volatile int		m_loop_status;
	wxCriticalSection	m_mutex_server;
};

#endif
