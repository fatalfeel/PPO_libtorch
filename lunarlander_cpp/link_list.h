#ifndef	LINK_LIST_H
#define	LINK_LIST_H

typedef struct _NODE_SOCKET
{
	unsigned long long		magic; //put first
	unsigned long long		lasttick;
	int						devid;
	int 					socketfd;
	int						recv_limit;
	unsigned int			recv_second;
	unsigned int			step;
	unsigned int			datasize;
	unsigned int			buf_pos;
	char*					buf_data;

	struct _NODE_SOCKET*	prev;
	struct _NODE_SOCKET*	next;
}NODE_SOCKET;

extern unsigned long long GetNowTick();

extern NODE_SOCKET* List_NetGetFirstNode();
extern NODE_SOCKET* List_NetGetLastNode();

extern void List_NetAddFront(int socketfd);
extern void List_NetAddRear(int socketfd);
extern void List_NetAddAssign(int socketfd, int nth);

extern void List_NetDeleteFront();
extern void List_NetDeleteRear();
extern void List_NetDeleteAssign(int nth);
extern void List_NetDeletebyNode(NODE_SOCKET* node);
extern void List_NetDeletebySocket(int socketfd);
extern void List_NetDeletebyDeviceId(int devid);
extern void List_NetDeleteAll();

extern NODE_SOCKET*		List_NetFindNodebyNth(int nth);
extern NODE_SOCKET*		List_NetFindNodebySocket(int socketfd);
extern NODE_SOCKET*		List_NetFindNodebyDeviceId(int devid);
extern int				List_NetLength();

#endif
