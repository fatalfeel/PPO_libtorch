#ifndef		SERVER_CMD_H
#define		SERVER_CMD_H

//step flow
/*Server Accept Client
   ->
   Check Magic 64bits
   ->
   Get Data Size
   ->
   buffer 32bits devid + xml data
   ->
   Check every IDLE connection
*/

//#define SERVER_BLUETOOTH

#define SERVER_PORT			9600 							//If SERVER_BLUETOOTH defined, set SERVER_PORT to 1
#define MAGIC_START 		0xFFFFFFFF12345678 	//TEL
#define MAGIC_START_REVERT	0x78563412FFFFFFFF
#define START_BITS 			0x01020304
#define START_BITS_REVERT	0x04030201
#define	MAX_LIMITBAND		(400*1024)			//bandwidth in one second
#define	MAX_BUFFERSIZE		(2*MAX_LIMITBAND)	//buffersize need >= limitband

enum
{
	STEP_MAGIC 		= 0x0001,
	STEP_SIZE 		= 0x0002,
	STEP_BUFDATA	= 0x0003	//process parameter of command
};

enum
{
	SOCKET_ADD			= 0x0001,
	SOCKET_DELETE		= 0x0002,
	SOCKET_DELETEALL	= 0x0003,
	SOCKET_DEVICEID		= 0x0005,
	SOCKET_BROADCAST	= 0x0006
};

#endif
