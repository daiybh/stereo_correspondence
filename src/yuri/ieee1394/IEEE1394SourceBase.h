/*!
 * @file 		IEEE1394InputBase.h
 * @author 		Zdenek Travnicek
 * @date 		29.5.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef IEEE1394INPUTBASE_H_
#define IEEE1394INPUTBASE_H_

#include <yuri/core/thread/IOThread.h>
#include <vector>
#include <libiec61883/iec61883.h>
namespace yuri {

namespace ieee1394 {
struct ieee1394_camera_info {
	int port;
	int node;
	uint64_t guid;
	std::string label;
};

class IEEE1394SourceBase: public core::IOThread
{
public:
	IEEE1394SourceBase(const  log::Log &log_, core::pwThreadBase parent, nodeid_t node=0, int port = 0, uint64_t guid=-1, const std::string& id="IEEE1394");
	virtual ~IEEE1394SourceBase() noexcept;
	virtual void run();
	bool initialized() { return device_ready; }
	int get_next_frame();
	//static int getNumberOfNodes(int port=0, int *local=0);
	//static int getNumberOfPorts();
	static int enumerateDevices(std::vector<ieee1394_camera_info> &devices);
protected:
	//static int receive_frame (unsigned char *data, int len, unsigned int complete, void *callback_data);
	//int process_frame(unsigned char *data, int len, int complete);
	virtual bool start_receiving() = 0;
	virtual bool stop_receiving() = 0;
	//static int busResetHandler(raw1394handle_t handle,  unsigned int generation);
	//void virtual setGeneration(unsigned int generation);
	virtual nodeid_t findNodeByGuid(raw1394handle_t handle, uint64_t guid);
	protected:
	raw1394handle_t handle;
	nodeid_t node;
	int channel,port;
	int oplug, iplug, bandwidth;
	bool device_ready;
};

}

}
#endif /* IEEE1394INPUTBASE_H_ */
