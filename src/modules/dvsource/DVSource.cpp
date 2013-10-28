/*!
 * @file 		DVSource.cpp
 * @author 		Zdenek Travnicek
 * @date 		16.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "DVSource.h"
#include "yuri/core/Module.h"
namespace yuri {

namespace ieee1394 {

REGISTER("dvsource",DVSource)

core::pIOThread DVSource::generate(log::Log &_log,core::pwThreadBase parent,core::Parameters& parameters)
{
	shared_ptr<DVSource> dv(new DVSource(_log,parent,
			parameters["node"].get<unsigned>(),
			parameters["port"].get<int>(),
			parameters["guid"].get<int64_t>()));
	return dv;
}
core::pParameters DVSource::configure()
{
	core::pParameters p (new core::Parameters());
	(*p)["node"]=0;
	(*p)["port"]=0;
	(*p)["guid"]=0;
	return p;
}

DVSource::DVSource(log::Log &log_,core::pwThreadBase parent, nodeid_t node, int port,
		int64_t guid):IEEE1394SourceBase(log_,parent,node,port,guid,"DVSource")
{
}

DVSource::~DVSource() {
}



int DVSource::receive_frame(unsigned char*data, int len, int complete, void *source)
{
	return ((DVSource*)source)->process_frame(data,len,complete);
}

int DVSource::process_frame(unsigned char *data, int length, int complete)
{
	log[log::verbose_debug] << "Received " << (complete?"":"in") << "complete frame with size " << length << " B" << std::endl;
	if (complete) {
		if (out[0]) {
			core::pBasicFrame frame = allocate_frame_from_memory(reinterpret_cast<yuri::ubyte_t*>(data),length);
			if (analyze_frame(frame))
				push_raw_video_frame(0,frame);

			log[log::verbose_debug] << "Sending frame" << std::endl;
		}
		else {
			log[log::warning] << "Received frame and don't have pipe to put it into, throwing away" << std::endl;
		}
	}
	return 0;
}

bool DVSource::start_receiving()
{
	frame = iec61883_dv_fb_init (handle, receive_frame, (void *)this);
	if (!frame) return false;
	log[log::debug] << "Starting to receive" << std::endl;
	if (iec61883_dv_fb_start (frame, channel)) return false;
	log[log::debug] << "Receiving" << std::endl;
	return true;
}

bool DVSource::stop_receiving()
{
	if (frame) iec61883_dv_fb_close (frame);
	return true;
}

#define BLOCKBYTE(block,byte) (80*(block)+byte)

bool DVSource::analyze_frame(core::pBasicFrame frame)
{
	//shared_array<yuri::ubyte_t> data = (*frame)[0].data;
	const yuri::ubyte_t * data = PLANE_RAW_DATA(frame,0);
	// First, lets check the header
	if (data[0]!=0x1f) {
		log[log::debug] << "This is not a valid DV frame! Expected magic 0x1f and got " << std::hex << data[0] << std::dec <<std::endl;
		return false;
	}
	log[log::verbose_debug] << "DV magic found" << std::endl;
	yuri::size_t fields=0;
	uint t = data[BLOCKBYTE(0,3)]&0xff;
	switch (t) {
		case 0x3f: log[log::verbose_debug] << "Found 60 field variant"<< std::endl;
				fields = 60;
				break;
		case 0xbf: log[log::verbose_debug] << "Found 50 field variant"<< std::endl;
				fields = 50;
				break;
		default: log[log::debug] << "Unknown number of fields " << std::hex << t<< std::endl;
			return false;
	}
	t = data[BLOCKBYTE(0,4)]&0x03;
//	bool iec;
	switch (t) {
		case 0: log[log::verbose_debug] << "Found IEC61834"<< std::endl;
//			iec = true;
			break;
		case 1: log[log::verbose_debug] << "Found SMPTE"<< std::endl;
//			iec = false;
			break;
		default:
			log[log::debug] << "track application id not recognized" << std::endl;
			return false;
	}
	int stype = data[80*5 + 48 + 3] & 0x1f;
	yuri::size_t width = 0;
	yuri::size_t height = 0;
	switch (stype) {
		case 0: log[log::verbose_debug] << "DVCAM, DVPRO" << std::endl;
			if (fields==50) {
				log[log::verbose_debug] << "625/PAL" << std::endl;
				width = 720;
				height = 576;
			} else {
				log[log::verbose_debug] << "525/NTSC" << std::endl;
				width = 720;
				height = 480;
			}
		break;
		case 4: log[log::verbose_debug] << "DVPCRO50" << std::endl;
			if (fields ==50) {
				log[log::verbose_debug] << "625/PAL 50Mbps" << std::endl;
				width = 720;
				height = 576;
			} else {
				log[log::verbose_debug] << "525/NTSC 50Mbps" << std::endl;
				width = 720;
				height = 480;
			}
		break;
		case 0x14: log[log::verbose_debug] << "1080i50 100 Mbps" << std::endl;
			if (fields ==50) {
				log[log::verbose_debug] << "625/PAL 50Mbps" << std::endl;
				width = 1440;
				height = 1080;
			} else {
				log[log::verbose_debug] << "1080i60 100 Mbps" << std::endl;
				width = 1280;
				height = 1080;
			}break;
		case 0x18: log[log::verbose_debug] << "DVCPRO HD 720p" << std::endl;
			if (fields ==50) {
				log[log::verbose_debug] << "720p50 100 Mbps" << std::endl;
				width = 960;
				height = 720;
			} else {
				log[log::verbose_debug] << "720p60 100 Mbps" << std::endl;
				width = 960;
				height = 720;
			}break;
		default: log[log::debug] << "Unknown signal type" << std::endl;
		return false;
	}
	log[log::verbose_debug] << "Setting DV " << width << "x" << height << std::endl;
	frame->set_parameters(YURI_VIDEO_DV, width, height);
	return true;
}


}

}


