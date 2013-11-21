/*!
 * @file 		DeltaInput.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "DeltaInput.h"
#include "yuri/core/Module.h"
#include <boost/assign.hpp>

namespace yuri {
namespace delta {

REGISTER("delta_input",DeltaInput)

IO_THREAD_GENERATOR(DeltaInput)

using namespace yuri::log;

core::pParameters DeltaInput::configure()
{
	core::pParameters p = core::IOThread::configure();
	p->set_description("DeltaInputcolor conversion module.");
//	(*p)["format"]["Output format"]=std::string("RGB24");
	p->set_max_pipes(1,1);
	return p;
}


DeltaInput::DeltaInput(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("deltainput")),format(YURI_FMT_RGB24),delta_handle_(0)
{
	IO_THREAD_INIT("DeltaInput")


	ULONG lib_version;
	ULONG boards;
	throw_call(VHD_GetApiInfo(&lib_version,&boards), "Failed to query driver version");
	log[log::info] << "Found driver version " << (lib_version>>24) << "." << ((lib_version>>16)&0xFF) << "." << (lib_version&0xFFFF);
	if (!boards) throw std::runtime_error("No boards found");
	log[log::info] << "Found " << boards << " boards";
	throw_call(VHD_OpenBoardHandle(0,&delta_handle_,0,0),"Failed to open the board");

}
namespace {
std::map<ULONG,std::string> chan_types = boost::assign::map_list_of<ULONG, std::string>
(VHD_CHNTYPE_SDSDI, "SDI")
(VHD_CHNTYPE_HDSDI, "HDSDI")
(VHD_CHNTYPE_3GSDI,"3GSDI");
}
void DeltaInput::run()
{
	IO_THREAD_PRE_RUN
	HANDLE stream=0, slot=0;
	try {
		ULONG chan_type;
		ULONG             clk;
		ULONG stat = 0;
		ULONG video_std;
		ULONG slots, dropped;
		VHD_GetBoardProperty(delta_handle_, VHD_CORE_BP_RX0_TYPE, &chan_type);
		log[log::info] << "Channel type: " << chan_types[chan_type];
		VHD_SetBoardProperty(delta_handle_,VHD_CORE_BP_BYPASS_RELAY_0,FALSE);
		while(!(stat& VHD_CORE_RXSTS_UNLOCKED)) {
			VHD_GetBoardProperty(delta_handle_, VHD_CORE_BP_RX0_STATUS, &stat);

			sleep(latency);
			if (!still_running()) throw std::runtime_error("Quit before lock was acquired");
		}
		throw_call(VHD_GetBoardProperty(delta_handle_,VHD_SDI_BP_RX0_CLOCK_DIV,&clk), "Failed to get board property");
		log[log::info] << ((clk==VHD_CLOCKDIV_1)?"Normal (EU) system":"Weird american system");

		throw_call(VHD_OpenStreamHandle(delta_handle_,VHD_ST_RX0,VHD_SDI_STPROC_DISJOINED_VIDEO,0,&stream,0),"Failed to get stream handle");
		throw_call(VHD_GetStreamProperty(stream,VHD_SDI_SP_VIDEO_STANDARD,&video_std),"Failed to get video standard");
		log[log::info] << "Receiving video standard " << video_std;
		VHD_SetStreamProperty(stream,VHD_SDI_SP_VIDEO_STANDARD,video_std);
		throw_call(VHD_StartStream(stream),"Failed to start stream");

		BYTE *buff;
		ULONG buff_size;
		while(still_running()) {
			VHD_LockSlotHandle(stream,&slot);

			throw_call(VHD_GetSlotBuffer(slot,VHD_SDI_BT_VIDEO,&buff,&buff_size),"Failed to get slot data");

			core::pBasicFrame frame = allocate_frame_from_memory(buff, buff_size);

			push_video_frame(0,frame,YURI_FMT_RGB24, 1920, 1080);

			VHD_UnlockSlotHandle(slot);

			VHD_GetStreamProperty(stream,VHD_CORE_SP_SLOTS_COUNT,&slots);
			VHD_GetStreamProperty(stream,VHD_CORE_SP_SLOTS_DROPPED,&dropped);
		}

	}
	catch (std::exception& e) {
		log[log::fatal] << "Error: " << e.what();

	}
	if (slot) {
		VHD_UnlockSlotHandle(slot);
	}
	if (stream) {
		VHD_StopStream(stream);
		VHD_CloseStreamHandle(stream);
	}
	IO_THREAD_POST_RUN
}
void DeltaInput::throw_call(ULONG res, std::string msg)
{
	if (res != VHDERR_NOERROR) throw std::runtime_error(msg);
}
DeltaInput::~DeltaInput()
{
}

bool DeltaInput::set_param(const core::Parameter& param)
{
//	if (param.name =="format") {
//		format = core::BasicPipe::get_format_from_string(param.get<std::string>());
//	} else
	return core::IOThread::set_param(param);
//	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
