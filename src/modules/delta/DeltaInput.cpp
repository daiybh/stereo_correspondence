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
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri {
namespace delta {

//REGISTER("delta_input",DeltaInput)

MODULE_REGISTRATION_BEGIN("delta")
                REGISTER_IOTHREAD("delta_input",DeltaInput)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(DeltaInput)

using namespace yuri::log;

core::Parameters DeltaInput::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("DeltaInputcolor conversion module.");
//	(*p)["format"]["Output format"]=std::string("RGB24");
//	p->set_max_pipes(1,1);
	return p;
}


DeltaInput::DeltaInput(log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("deltainput")),format(core::raw_format::rgb24),delta_handle_(0)
{
	IOTHREAD_INIT(parameters)
	set_latency(1_ms);

	ULONG lib_version;
	ULONG boards;
	throw_call(VHD_GetApiInfo(&lib_version,&boards), "Failed to query driver version");
	log[log::info] << "Found driver version " << (lib_version>>24) << "." << ((lib_version>>16)&0xFF) << "." << (lib_version&0xFFFF);
	if (!boards) throw std::runtime_error("No boards found");
	log[log::info] << "Found " << boards << " boards";
	throw_call(VHD_OpenBoardHandle(0,&delta_handle_,0,0),"Failed to open the board");


}
namespace {
std::map<ULONG,std::string> chan_types = {
{VHD_CHNTYPE_SDSDI, "SDI"},
{VHD_CHNTYPE_HDSDI, "HDSDI"},
{VHD_CHNTYPE_3GSDI,"3GSDI"}};
}
void DeltaInput::run()
{
//	IO_THREAD_PRE_RUN

	HANDLE stream=0, slot=0;
	try {
		ULONG chan_type;
		ULONG             clk;
		ULONG stat = 0;
		ULONG video_std;
//		ULONG slots, dropped;
		VHD_GetBoardProperty(delta_handle_, VHD_CORE_BP_RX0_TYPE, &chan_type);
		log[log::info] << "Channel type: " << chan_types[chan_type];
		VHD_SetBoardProperty(delta_handle_,VHD_CORE_BP_BYPASS_RELAY_0,FALSE);

		VHD_GetBoardProperty(delta_handle_, VHD_CORE_BP_RX0_STATUS, &stat);

		// Wait for incomming signal
		while(stat& VHD_CORE_RXSTS_UNLOCKED) {
			if (!still_running()) throw std::runtime_error("Quit before lock was acquired");

			if (VHD_GetBoardProperty(delta_handle_, VHD_CORE_BP_RX0_STATUS, &stat)!=VHDERR_NOERROR) continue;
			log[log::info]<<stat;
			sleep(get_latency());
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

			// This shouldn't be throw_call probably...
			throw_call(VHD_GetSlotBuffer(slot,VHD_SDI_BT_VIDEO,&buff,&buff_size),"Failed to get slot data");
			log[log::info] << "Reading " << buff_size << " bytes";
			//core::pBasicFrame frame = allocate_frame_from_memory(buff, buff_size);
			auto frame = core::RawVideoFrame::create_empty(core::raw_format::uyvy422, {1920, 1080}, reinterpret_cast<uint8_t*>(buff), buff_size);

			push_frame(0,frame);//, core::raw_format::rgb24, 1920, 1080);

			VHD_UnlockSlotHandle(slot);

//			VHD_GetStreamProperty(stream,VHD_CORE_SP_SLOTS_COUNT,&slots);
//			VHD_GetStreamProperty(stream,VHD_CORE_SP_SLOTS_DROPPED,&dropped);
		}

	}
	catch (std::exception& e) {
		log[log::fatal] << "Error: " << e.what();
		request_end();

	}
	if (slot) {
		VHD_UnlockSlotHandle(slot);
	}
	if (stream) {
		VHD_StopStream(stream);
		VHD_CloseStreamHandle(stream);
	}
//	IO_THREAD_POST_RUN
}
void DeltaInput::throw_call(ULONG res, std::string msg)
{
	if (res != VHDERR_NOERROR) throw std::runtime_error(msg);
}
DeltaInput::~DeltaInput() noexcept
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
