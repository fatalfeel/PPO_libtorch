#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <sys/socket.h>
#include "server_cmd.h"
#include "link_list.h"

/*typedef struct _TIMESPEC 
{ 
	long tv_sec; 
	long tv_nsec; 
}TIMESPEC ;    //header part

int clock_gettime(int, TIMESPEC* spec)      //C-file part
{ 
	long long wintime; 
	GetSystemTimeAsFileTime((FILETIME*)&wintime);
	wintime      -= 116444736000000000LL;           //1jan1601 to 1jan1970
	spec->tv_sec  = wintime / 10000000LL;           //seconds
	spec->tv_nsec = wintime % 10000000LL *100;      //nano-seconds
	
	return 0;
}*/

unsigned long long GetNowTick()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);

	return (unsigned long long)ts.tv_sec*1000000000 + ts.tv_nsec;
}

//////////////////////////////////////////////////////////////////
static	NODE_SOCKET* s_sockethead 	= NULL;
static 	NODE_SOCKET* s_sockettail	= NULL;

NODE_SOCKET* List_NetGetFirstNode()
{
	return s_sockethead;
}

NODE_SOCKET* List_NetGetLastNode()
{
	return s_sockettail;
}

/////////add node/////////
void List_NetAddFront(int socketfd)
{
	NODE_SOCKET* newnode;
	printf("%s: value %d added\n", __FUNCTION__, socketfd);

	if ( newnode = (NODE_SOCKET*)malloc(sizeof(struct _NODE_SOCKET)) )
	{
		newnode->socketfd 	= socketfd;
		newnode->step		= STEP_MAGIC;
		newnode->lasttick	= GetNowTick();
		newnode->magic		= 0;
		newnode->recv_limit	= MAX_BUFFERSIZE;
		newnode->recv_second= MAX_LIMITBAND;
		newnode->datasize	= 0;
		newnode->devid		= 0;
		newnode->buf_pos	= 0;
		newnode->buf_data	= NULL;
		newnode->next = NULL;
		newnode->prev = NULL;
	}
	else
	{
		printf("unable to allocate memory \n");
		return;
	}

	if (s_sockethead == NULL)
	{
		assert (s_sockettail == NULL);
		s_sockethead = s_sockettail = newnode;
	}
	else
	{
		newnode->next = s_sockethead;
		s_sockethead->prev = newnode;
		s_sockethead = newnode;
	}
}

void List_NetAddRear(int socketfd)
{
	NODE_SOCKET* newnode;
	printf("%s: socket %d added\n", __FUNCTION__, socketfd);

	if ( newnode = (NODE_SOCKET*)malloc(sizeof(struct _NODE_SOCKET)) )
	{
		newnode->socketfd 	= socketfd;
		newnode->step		= STEP_MAGIC;
		newnode->lasttick	= GetNowTick();
		newnode->magic		= 0;
		newnode->recv_limit	= MAX_BUFFERSIZE;
		newnode->recv_second= MAX_LIMITBAND;
		newnode->datasize	= 0;
		newnode->devid		= 0;
		newnode->buf_pos	= 0;
		newnode->buf_data	= NULL;
		newnode->next = NULL;
		newnode->prev = NULL;
	}
	else
	{
		printf("unable to allocate memory\n");
		return;
	}

	if (s_sockettail == NULL)
	{
		assert (s_sockettail == NULL);
		s_sockethead = s_sockettail = newnode;
	}
	else
	{
		newnode->prev = s_sockettail;
		s_sockettail->next = newnode;
		s_sockettail = newnode;
	}
}

void List_NetAddAssign(int socketfd, int nth)
{
	int 			len = List_NetLength();
	NODE_SOCKET*	newnode;
	NODE_SOCKET*	cur = s_sockethead;

	printf("%s: value %d nth %d added\n", __FUNCTION__, socketfd, nth);

	if (nth < 0 || nth > len+1)
	{
		printf("Invalid nth \n");
		return;
	}

	if ( newnode = (NODE_SOCKET*)malloc(sizeof(struct _NODE_SOCKET)) )
	{
		newnode->socketfd 	= socketfd;
		newnode->step		= STEP_MAGIC;
		newnode->lasttick	= GetNowTick();
		newnode->magic		= 0;
		newnode->recv_limit	= MAX_BUFFERSIZE;
		newnode->recv_second= MAX_LIMITBAND;
		newnode->datasize	= 0;
		newnode->devid		= 0;
		newnode->buf_pos	= 0;
		newnode->buf_data	= NULL;
		newnode->next = NULL;
		newnode->prev = NULL;
	}
	else
	{
		printf("unable to allocate memory \n");
		return;
	}

	if (nth == 0)
	{
		if (s_sockethead == NULL)
		{
			assert (s_sockettail == NULL);
			s_sockethead = s_sockettail = newnode;
			printf("added \n");
			return;
		}

		newnode->next = s_sockethead;
		s_sockethead->prev = newnode;
		s_sockethead = newnode;
		printf("added \n");
		return;
	}

	if (nth == len+1)
	{
		s_sockettail->next = newnode;
		newnode->prev = s_sockettail;
		s_sockettail = newnode;
		printf("added \n");
		return;
	}

	--nth;
	while (--nth)
	{
		cur = cur->next;
	}

	newnode->prev = cur;
	newnode->next = cur->next;
	cur->next->prev = newnode;
	cur->next = newnode;
}

/////////delete node/////////
void List_NetDeleteFront()
{
	printf("%s: ", __FUNCTION__);
	NODE_SOCKET *temp = s_sockethead;

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list\n");
		return;
	}

	if (s_sockethead == s_sockettail)
	{
		s_sockethead = s_sockettail = NULL;
	}
	else
	{
		s_sockethead = s_sockethead->next;
		s_sockethead->prev = NULL;
	}

	printf("%d deleted \n", temp->socketfd);

	shutdown(temp->socketfd, SHUT_RDWR);
	close(temp->socketfd);
	if( temp->buf_data )
	{
		free(temp->buf_data);
		temp->buf_data = NULL;
	}
	free(temp);
}

void List_NetDeleteRear()
{
	printf("%s: ", __FUNCTION__);
	NODE_SOCKET *temp = s_sockettail;

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list\n");
		return;
	}

	if (s_sockethead == s_sockettail)
	{
		s_sockethead = s_sockettail = NULL;
	}
	else
	{
		s_sockettail = s_sockettail->prev;
		s_sockettail->next = NULL;
	}

	printf("%d deleted \n", temp->socketfd);

	shutdown(temp->socketfd, SHUT_RDWR);
	close(temp->socketfd);
	if( temp->buf_data )
	{
		free(temp->buf_data);
		temp->buf_data = NULL;
	}
	free(temp);
}

void List_NetDeleteAssign(int nth)
{
	int len				= List_NetLength();
	NODE_SOCKET* cur 	= s_sockethead;
	NODE_SOCKET* temp	= cur;

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list\n");
		return;
	}

	if ( nth < 0 || nth > len )
	{
		printf("%d Invalid nth\n", nth);
		return;
	}

	if (nth == 0)
	{
		if (s_sockethead == s_sockettail)
		{
			s_sockethead = s_sockettail = NULL;
			printf(" %d deleted\n", temp->socketfd);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		s_sockethead = s_sockethead->next;
		s_sockethead->prev = NULL;
		printf(" %d deleted\n", temp->socketfd);

		shutdown(temp->socketfd, SHUT_RDWR);
		close(temp->socketfd);
		if( temp->buf_data )
		{
			free(temp->buf_data);
			temp->buf_data = NULL;
		}
		free(temp);
		return;
	}

	if (nth == len)
	{
		if (s_sockethead == s_sockettail)
		{
			s_sockethead = s_sockettail = NULL;
			printf(" %d deleted\n", temp->socketfd);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		temp = s_sockettail;
		s_sockettail = s_sockettail->prev;
		s_sockettail->next = NULL;

		printf(" %d deleted\n", temp->socketfd);

		shutdown(temp->socketfd, SHUT_RDWR);
		close(temp->socketfd);
		if( temp->buf_data )
		{
			free(temp->buf_data);
			temp->buf_data = NULL;
		}
		free(temp);
		return;
	}

	--nth;
	while (--nth)
		cur = cur->next;

	temp = cur->next;
	cur->next = cur->next->next;
	cur->next->prev = cur;

	printf(" %d deleted\n", temp->socketfd);

	shutdown(temp->socketfd, SHUT_RDWR);
	close(temp->socketfd);
	if( temp->buf_data )
	{
		free(temp->buf_data);
		temp->buf_data = NULL;
	}
	free(temp);
}

void List_NetDeletebyNode(NODE_SOCKET* node)
{
	NODE_SOCKET *cur = s_sockethead;
	NODE_SOCKET *temp = NULL;

	printf("%s: ", __FUNCTION__);

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list\n");
		return;
	}

	if ( node == s_sockethead )
	{
		temp = s_sockethead;

		if (s_sockethead == s_sockettail)
		{
			s_sockethead = s_sockettail = NULL;
			printf("%d deleted\n\n", temp->socketfd);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		s_sockethead = s_sockethead->next;
		s_sockethead->prev = NULL;
		printf("%d deleted\n\n", temp->socketfd);

		shutdown(temp->socketfd, SHUT_RDWR);
		close(temp->socketfd);
		if( temp->buf_data )
		{
			free(temp->buf_data);
			temp->buf_data = NULL;
		}
		free(temp);
		return;
	}

	while (cur != NULL)
	{
		if (cur == node)
		{
			temp = cur;
			if (cur == s_sockettail)
			{
				s_sockettail = s_sockettail->prev;
				s_sockettail->next = NULL;
			}
			else
			{
				cur->prev->next = cur->next;
				cur->next->prev = cur->prev;
			}

			printf("%d deleted\n\n", temp->socketfd);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		cur = cur->next;
	}

	//printf("Can not find node address %x\n", node);
}

void List_NetDeletebySocket(int socketfd)
{
	NODE_SOCKET *cur = s_sockethead;
	NODE_SOCKET *temp = NULL;

	printf("%s: ", __FUNCTION__);

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list \n");
		return;
	}

	if ( socketfd == s_sockethead->socketfd )
	{
		temp = s_sockethead;

		if (s_sockethead == s_sockettail)
		{
			s_sockethead = s_sockettail = NULL;

			printf("%d deleted\n", temp->socketfd);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		s_sockethead = s_sockethead->next;
		s_sockethead->prev = NULL;
		printf("%d deleted\n", temp->socketfd);

		shutdown(temp->socketfd, SHUT_RDWR);
		close(temp->socketfd);
		if( temp->buf_data )
		{
			free(temp->buf_data);
			temp->buf_data = NULL;
		}
		free(temp);
		return;
	}

	while (cur != NULL)
	{
		if (cur->socketfd == socketfd)
		{
			temp = cur;
			if (cur == s_sockettail)
			{
				s_sockettail = s_sockettail->prev;
				s_sockettail->next = NULL;
			}
			else
			{
				cur->prev->next = cur->next;
				cur->next->prev = cur->prev;
			}

			printf("%d deleted\n", temp->socketfd);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		cur = cur->next;
	}

	printf("Can not find %d\n", socketfd);
}

void List_NetDeletebyDeviceId(int devid)
{
	NODE_SOCKET *cur = s_sockethead;
	NODE_SOCKET *temp = NULL;

	printf("%s: ", __FUNCTION__);

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list \n");
		return;
	}

	if ( devid == s_sockethead->devid )
	{
		temp = s_sockethead;

		if (s_sockethead == s_sockettail)
		{
			s_sockethead = s_sockettail = NULL;

			printf("%d deleted\n", temp->devid);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		s_sockethead = s_sockethead->next;
		s_sockethead->prev = NULL;
		printf("%d deleted\n", temp->devid);

		shutdown(temp->socketfd, SHUT_RDWR);
		close(temp->socketfd);
		if( temp->buf_data )
		{
			free(temp->buf_data);
			temp->buf_data = NULL;
		}
		free(temp);
		return;
	}

	while (cur != NULL)
	{
		if (cur->devid == devid)
		{
			temp = cur;
			if (cur == s_sockettail)
			{
				s_sockettail = s_sockettail->prev;
				s_sockettail->next = NULL;
			}
			else
			{
				cur->prev->next = cur->next;
				cur->next->prev = cur->prev;
			}

			printf("%d deleted\n", temp->devid);

			shutdown(temp->socketfd, SHUT_RDWR);
			close(temp->socketfd);
			if( temp->buf_data )
			{
				free(temp->buf_data);
				temp->buf_data = NULL;
			}
			free(temp);
			return;
		}

		cur = cur->next;
	}

	printf("Can not find %d\n", devid);
}

void List_NetDeleteAll()
{
	printf("%s: ", __FUNCTION__);
	NODE_SOCKET *cur = s_sockethead;
	NODE_SOCKET *temp;

	while (cur != NULL)
	{
		temp = cur;
		cur = cur->next;

		shutdown(temp->socketfd, SHUT_RDWR);
		close(temp->socketfd);
		if( temp->buf_data )
		{
			free(temp->buf_data);
			temp->buf_data = NULL;
		}
		free(temp);
	}

	s_sockethead = s_sockettail = NULL;

	printf("Done \n");
}

/////////find node/////////
NODE_SOCKET* List_NetFindNodebyNth(int nth)
{
	printf("%s: find %d\n", __FUNCTION__, nth);
	int						k;
	NODE_SOCKET*	cur = s_sockethead;

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list \n");
		return NULL;
	}

	k = 0;
	while (cur != NULL)
	{
		if ( nth == k )
			return cur;

		k++;
		cur = cur->next;
	}

	printf("Find node nth %02d is exist\n", nth);

	return NULL;
}

NODE_SOCKET* List_NetFindNodebySocket(int socketfd)
{
	printf("%s: find %d\n", __FUNCTION__, socketfd);
	NODE_SOCKET *cur = s_sockethead;

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list \n");
		return NULL;
	}

	if (s_sockethead->socketfd == socketfd)
		return s_sockethead;

	while (cur != NULL)
	{
		if (cur->socketfd == socketfd)
			return cur;

		cur = cur->next;
	}

	printf("Find socket %d is not exist\n", socketfd);

	return NULL;
}

NODE_SOCKET* List_NetFindNodebyDeviceId(int devid)
{
	printf("%s: find %d\n", __FUNCTION__, devid);
	NODE_SOCKET *cur = s_sockethead;

	if (s_sockethead == NULL)
	{
		assert(s_sockettail == NULL);
		printf("Empty list \n");
		return NULL;
	}

	if (s_sockethead->devid == devid)
		return s_sockethead;

	while (cur != NULL)
	{
		if (cur->devid == devid)
			return cur;

		cur = cur->next;
	}

	printf("Find socket %d is not exist\n", devid);

	return NULL;
}
/////////node length/////////
int List_NetLength()
{
	int len = 0;
	NODE_SOCKET *cur = s_sockethead;

	while(cur != NULL)
	{
		len++;
		cur = cur->next;
	}

	return len;
}

