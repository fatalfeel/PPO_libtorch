#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <arpa/inet.h>
#include "server_cmd.h"
#include "link_list.h"
#include "wxlocker.h"
#include "servermessage.h"

static ServerMessage* s_servermessage_instance = NULL;

ServerMessage* ServerMessage::Get()
{
	if (!s_servermessage_instance)
		s_servermessage_instance = new ServerMessage();

	return s_servermessage_instance;
}

void ServerMessage::Free()
{
	if( s_servermessage_instance )
	{
		delete s_servermessage_instance;
		s_servermessage_instance = NULL;
	}
}

ServerMessage::ServerMessage()
{
	m_serversocket = -1;
}

ServerMessage::~ServerMessage()
{
	//stop server accept to wait
	if( m_serversocket != -1 )
	{
		shutdown(m_serversocket, SHUT_RDWR);
		close(m_serversocket);
	}
	
	m_thread_work = 0;
	while( m_loop_status )
	{
		usleep(200000); //200ms
	}
}

void ServerMessage::SocketProcess(int type, NODE_SOCKET* process_ptr)
{
	wxCriticalLocker lock(m_mutex_server);

	NODE_SOCKET* local_ptr;

	switch(type)
	{
		case SOCKET_ADD:
			List_NetAddRear(process_ptr->socketfd);
			break;

		case SOCKET_DELETE:
			List_NetDeletebyNode(process_ptr);
			break;

		case SOCKET_DELETEALL:
			List_NetDeleteAll();
			break;

		case SOCKET_DEVICEID:
			local_ptr = List_NetFindNodebyDeviceId(process_ptr->devid);
			if( local_ptr )
			{
				send(local_ptr->socketfd, (char*)&process_ptr->magic, 8, 0);
				send(local_ptr->socketfd, (char*)&process_ptr->datasize, 4, 0);
				send(local_ptr->socketfd, process_ptr->buf_data, process_ptr->datasize, 0);
			}
			break;

		case SOCKET_BROADCAST:
			local_ptr = List_NetGetFirstNode();
			while( local_ptr )
			{
				send(local_ptr->socketfd, (char*)&process_ptr->magic, 8, 0);
				send(local_ptr->socketfd, (char*)&process_ptr->datasize, 4, 0);
				send(local_ptr->socketfd, process_ptr->buf_data, process_ptr->datasize, 0);
				local_ptr = local_ptr->next;
			}
			break;
	}
}

void ServerMessage::InitServerSocket(int port)
{
	int 				err;
	int					opt_val;
	int					socketfd;
	struct sockaddr_in	server_addr = { 0 };

	socketfd = socket(AF_INET, SOCK_STREAM, 0);

	if (socketfd < 0)
		printf("Could not create server_addr socket: %s\n", strerror(errno));

	opt_val 						= 1;
	server_addr.sin_family 			= AF_INET;
	server_addr.sin_addr.s_addr		= htonl(INADDR_ANY);		//remote connect
	//server_addr.sin_addr.s_addr	= inet_addr("127.0.0.1");	//local connect
	server_addr.sin_port 			= htons(port);
	setsockopt(socketfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt_val, sizeof(opt_val));

	err = bind(socketfd, (struct sockaddr*) &server_addr, sizeof(server_addr));
	if (err < 0)
		printf("Could not bind server_addr socket: %s\n", strerror(errno));

	err = listen(socketfd, SOMAXCONN);

	if (err < 0)
		printf("Could not listen: %s\n", strerror(errno));

	m_serversocket = socketfd;
}

int ServerMessage::MagicPadCheck(unsigned long long* magic, unsigned long long onebyte)
{
	*magic	= (*magic<<8) | onebyte;

	if( *magic == MAGIC_START_REVERT )
		return true;

	return false;
}

//your job function here is a test
void ServerMessage::ExternalJob(NODE_SOCKET* node_ptr)
{
	node_ptr->magic	= MAGIC_START;
	
	//data
	memset(node_ptr->buf_data,0x77,node_ptr->datasize);
					
	WriteData(node_ptr);
}

//https://linux.die.net/man/2/send
void ServerMessage::WriteData(NODE_SOCKET* node_ptr)
{
	int	len;
	int	send_shift;
	int	send_remain;
	
	if( node_ptr->socketfd > 0 )
	{
		len = send(node_ptr->socketfd, (char*)&node_ptr->magic, sizeof(node_ptr->magic), 0);
		len = send(node_ptr->socketfd, (char*)&node_ptr->datasize, sizeof(node_ptr->datasize), 0);
		
		send_shift	= 0;
		send_remain	= node_ptr->datasize;
		while( send_remain > 0 )
		{
			len = send(node_ptr->socketfd, node_ptr->buf_data+send_shift, send_remain, 0);
	
			if( len >= 0 )
			{
				send_shift	+= len;
				send_remain	-= len;
			}
			else
			{
				break;
			}
		}
	}
}

//static void* ServerManyCome(void* param)
void* ServerMessage::ServerThreadLoop()
{
	unsigned char		onebyte;
	int 				acceptfd;
	int 				maxfd;
	int 				len;
	int					padcheck;
	unsigned int		recv_remain;
	unsigned int		recv_shift;
	unsigned long long	tickerband;
	unsigned long long	tickernow;
	unsigned long long	timeoutref;
	fd_set 				svset;
	fd_set 				rdset;
	socklen_t 			addrlen;
	struct sockaddr_in 	clientaddr;
	NODE_SOCKET			node_ext;
	NODE_SOCKET*		node_ptr;
	char*				recvbuf; //if array over than 1k use alloc

	m_thread_work 	= 1;
	m_loop_status	= 1;

	if (m_serversocket <= -1) //check socket
	{
		m_thread_work = 0;
		m_loop_status = 0;
		printf("Fail to allocate the server socket %d\n", m_serversocket);

		return 0;
	}

	tickerband	= GetNowTick();
	timeoutref	= 60 * 10000000LL; //60 sec
	recvbuf		= (char*)malloc(MAX_BUFFERSIZE);
	
	/* Keep track of the max file descriptor */
	maxfd = m_serversocket;
	printf("Accept / maxfd = %d\n\n", maxfd);

	//List_NetAddRear(m_serversocket);
	node_ext.socketfd = m_serversocket;
	SocketProcess(SOCKET_ADD, &node_ext);

	/* Error signal */
	//signal(SIGINT, close_sigint);

	/* Clear the reference set of socket */
	FD_ZERO(&svset);

	/* Add the server socket */
	FD_SET(m_serversocket, &svset);
	
	while( m_thread_work )
	{
		rdset = svset;
		//use select monitor all sockets of list
		if (select(maxfd+1, &rdset, NULL, NULL, NULL) == -1)
		{
			m_thread_work = 0; //exit while loop
			printf("Server select failure \n");
			//close_sigint(-1);
		}

		/////////New  client and data receive/////////
		node_ptr = List_NetGetFirstNode();
		while( node_ptr )
		{
			if (FD_ISSET(node_ptr->socketfd, &rdset))
			{
				if( node_ptr->socketfd == m_serversocket )
				{
					/* Handle new connections */
					addrlen = sizeof(clientaddr);
					memset(&clientaddr, 0, sizeof(clientaddr));
					acceptfd = accept(m_serversocket, (struct sockaddr*)&clientaddr, &addrlen);

					if (acceptfd <= -1)
					{
						printf("Server accept error \n");
					}
					else
					{
						FD_SET(acceptfd, &svset);

						if (acceptfd > maxfd)
						{
							/* Keep track of the maximum */
							maxfd = acceptfd;
						}

						//List_NetAddRear(acceptfd);
						node_ext.socketfd = acceptfd;
						SocketProcess(SOCKET_ADD, &node_ext);

						printf("New connection IP %s\n", inet_ntoa(clientaddr.sin_addr));
						printf("Maxfd = %d\n\n", maxfd);
					}
				}
				else
				{
					len = recv(node_ptr->socketfd, recvbuf, node_ptr->recv_limit, 0);
					
					node_ptr->recv_second  += len;
					node_ptr->lasttick		= GetNowTick();
					
					if (len > 0)
					{
						recv_remain = len;
						recv_shift	= 0;

						while( recv_remain > 0 )
						{
							onebyte		= recvbuf[recv_shift];
							padcheck	= MagicPadCheck(&node_ptr->magic, onebyte);
							
							recv_shift++;
							recv_remain = len - recv_shift;
							
							switch( node_ptr->step )
							{
								case STEP_MAGIC:
									if( padcheck )
									{
										node_ptr->step 		= STEP_SIZE;
										node_ptr->buf_pos 	= 0; //size of buffer position
									}
									break;

								case STEP_SIZE:
									if( padcheck )
									{
										node_ptr->step 		= STEP_SIZE;
										node_ptr->buf_pos 	= 0; //size of buffer position
										break;
									}

									*((unsigned char*)&node_ptr->datasize+node_ptr->buf_pos) = onebyte;
									node_ptr->buf_pos++;

									if( node_ptr->buf_pos >= sizeof(node_ptr->datasize) )
									{
										if( node_ptr->datasize <= 0 || node_ptr->datasize > MAX_BUFFERSIZE )
										{
											node_ptr->step		= STEP_MAGIC; //error size init
										}
										else
										{
											node_ptr->step 		= STEP_BUFDATA;
											node_ptr->buf_pos	= 0;
											node_ptr->buf_data	= (char*)malloc(node_ptr->datasize);
										}
									}
									break;

								case STEP_BUFDATA:
									if( padcheck )
									{
										node_ptr->step 		= STEP_SIZE;
										node_ptr->buf_pos 	= 0; //size of buffer position
										free(node_ptr->buf_data);
										node_ptr->buf_data = NULL;
										break;
									}

									node_ptr->buf_data[node_ptr->buf_pos]= onebyte;
									node_ptr->buf_pos++;

									if( node_ptr->buf_pos >= node_ptr->datasize )
									{
										ExternalJob(node_ptr);

										node_ptr->step = STEP_MAGIC;
										free(node_ptr->buf_data);
										node_ptr->buf_data 	= NULL;
									}
									break;
							}
						}//end while
					}
					else
					{
						FD_CLR(node_ptr->socketfd, &svset);
						if (node_ptr->socketfd >= maxfd)
						{
							/* Keep track of the maximum */
							maxfd--;
						}
												
						printf("Disconnect remove socket %d\n", node_ptr->socketfd);
						printf("Disconnect and maxfd = %d\n", maxfd);

						node_ext.next = node_ptr->next;
						SocketProcess(SOCKET_DELETE, node_ptr);
						node_ptr = &node_ext;
					}
				}
			}//if (FD_ISSET(node_ptr->socketfd, &rdset))
			
			node_ptr = node_ptr->next;
			/*::Sleep(10);*/ //don't do this, test except

		}//while(node)

		/////////bandwidth check in every second/////////
		tickernow = GetNowTick();
		if( tickernow - tickerband > 10000000LL )
		{
			tickerband	= tickernow;
			node_ptr	= List_NetGetFirstNode();
			while( node_ptr )
			{
				if( node_ptr->socketfd != m_serversocket )
				{
					if( node_ptr->recv_second > MAX_LIMITBAND )
					{
						if( node_ptr->recv_limit - 64 > 0 )
							node_ptr->recv_limit -= 64;
					}
					else if( node_ptr->recv_second < MAX_LIMITBAND )
					{
						if( node_ptr->recv_limit + 64 <= MAX_BUFFERSIZE )
							node_ptr->recv_limit += 64;
					}

					node_ptr->recv_second = 0;
				}
				
				node_ptr = node_ptr->next;
			}
		}

		/////////Timeout check/////////
		tickernow	= GetNowTick();
		node_ptr	= List_NetGetFirstNode();
		while( node_ptr )
		{
			if( node_ptr->socketfd != m_serversocket )
			{
				if( tickernow - node_ptr->lasttick > timeoutref )
				{
					FD_CLR(node_ptr->socketfd, &svset);
					if (node_ptr->socketfd >= maxfd)
					{
						maxfd--;
					}
					
					printf("Timeout remove socket %d\n", node_ptr->socketfd);
					printf("Timeout and maxfd is %d\n", maxfd);

					node_ext.next	= node_ptr->next;
					SocketProcess(SOCKET_DELETE, node_ptr);
					node_ptr		= &node_ext;
				}
			}
			node_ptr = node_ptr->next;
		}//while(node)
	}// while( s_server_work )

	SocketProcess(SOCKET_DELETEALL, NULL);
	
	free(recvbuf);
		
	m_loop_status = 0;
	
	return 0;
}

//static method
void* ServerMessage::ThreadInterface(void* param)
{
	((ServerMessage*)param)->ServerThreadLoop();
}

void ServerMessage::StartServerThread()
{
	void* threadfunc;

	pthread_t 		pid;
	pthread_attr_t 	pattr;


	pthread_attr_init(&pattr);
    pthread_attr_setdetachstate(&pattr, PTHREAD_CREATE_DETACHED);
    pthread_create(&pid, &pattr, &this->ThreadInterface, this); //static method
    //pthread_create(&pid, &pattr, (THREADFUNC_PTR)threadfunc, this);
    pthread_attr_destroy(&pattr);
}

/*int main(int argc, char **argv)
{
	//int	myloop;
	
	ServerMessage::Get()->InitServerSocket(SERVER_PORT);
	ServerMessage::Get()->StartServerThread();
    
	_getche();

	ServerMessage::Free();

	return 0;
}*/
