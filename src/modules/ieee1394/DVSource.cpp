/*!
 * @file 		DVSource.cpp
 * @author 		Zdenek Travnicek
 * @date 		16.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "DVSource.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/compressed_frame_types.h"

namespace yuri {

namespace ieee1394 {

core::pIOThread DVSource::generate(log::Log &_log,core::pwThreadBase parent, const core::Parameters& parameters)
{
	shared_ptr<DVSource> dv(new DVSource(_log,parent,
			parameters["node"].get<unsigned>(),
			parameters["port"].get<int>(),
			parameters["guid"].get<int64_t>()));
	return dv;
}
core::Parameters DVSource::configure()
{
	core::Parameters p = IEEE1394SourceBase::configure();
	p["node"]=0;
	p["port"]=0;
	p["guid"]=0;
	return p;
}

DVSource::DVSource(const log::Log &log_,core::pwThreadBase parent, nodeid_t node, int port,
	int64_t guid):IEEE1394SourceBase(log_,parent,node,port,guid,"DVSource")
{
}

DVSource::~DVSource() noexcept {
}



int DVSource::receive_frame(unsigned char*data, int len, int complete, void *source)
{
	return ((DVSource*)source)->process_frame(data,len,complete);
}

int DVSource::process_frame(unsigned char *data, int length, int complete)
{
	log[log::verbose_debug] << "Received " << (complete?"":"in") << "complete frame with size " << length << " B";
	if (complete) {
		core::pCompressedVideoFrame frame = core::CompressedVideoFrame::create_empty(core::compressed_frame::dv, resolution_t{720, 576}, reinterpret_cast<const uint8_t*>(data), length);
		if (analyze_frame(frame)) {
			push_frame(0,frame);
			log[log::verbose_debug] << "Sending frame";
		}
	}
	return 0;
}

bool DVSource::start_receiving()
{
	frame = iec61883_dv_fb_init (handle, receive_frame, (void *)this);
	if (!frame) return false;
	log[log::debug] << "Starting to receive";
	if (iec61883_dv_fb_start (frame, channel)) return false;
	log[log::debug] << "Receiving";
	return true;
}

bool DVSource::stop_receiving()
{
	if (frame) iec61883_dv_fb_close (frame);
	return true;
}

#define BLOCKBYTE(block,byte) (80*(block)+byte)

bool DVSource::analyze_frame(core::pCompressedVideoFrame& frame)
{
	//shared_array<uint8_t> data = (*frame)[0].data;
	const uint8_t * data = frame->data();
	// First, lets check the header
	if (data[0]!=0x1f) {
		log[log::debug] << "This is not a valid DV frame! Expected magic 0x1f and got " << std::hex << data[0] << std::dec;
		return false;
	}
	log[log::verbose_debug] << "DV magic found";
	yuri::size_t fields=0;
	uint t = data[BLOCKBYTE(0,3)]&0xff;
	switch (t) {
		case 0x3f: log[log::verbose_debug] << "Found 60 field variant";
				fields = 60;
				break;
		case 0xbf: log[log::verbose_debug] << "Found 50 field variant";
				fields = 50;
				break;
		default: log[log::debug] << "Unknown number of fields " << std::hex << t;
			return false;
	}
	t = data[BLOCKBYTE(0,4)]&0x03;
//	bool iec;
	switch (t) {
		case 0: log[log::verbose_debug] << "Found IEC61834";
//			iec = true;
			break;
		case 1: log[log::verbose_debug] << "Found SMPTE";
//			iec = false;
			break;
		default:
			log[log::debug] << "track application id not recognized";
			return false;
	}
	int stype = data[80*5 + 48 + 3] & 0x1f;
	resolution_t res = {0, 0};
	duration_t dur;
	switch (stype) {
		case 0: log[log::verbose_debug] << "DVCAM, DVPRO";
			if (fields==50) {
				log[log::verbose_debug] << "625/PAL";
				res = {720, 576};
				dur = 1_s / 25;
			} else {
				log[log::verbose_debug] << "525/NTSC";
				res = {720, 480};
				dur = 1_s / 30;
			}
		break;
		case 4: log[log::verbose_debug] << "DVPCRO50";
			if (fields ==50) {
				log[log::verbose_debug] << "625/PAL 50Mbps";
				res = {720, 576};
				dur = 1_s / 25;
			} else {
				log[log::verbose_debug] << "525/NTSC 50Mbps";
				res = {720, 480};
				dur = 1_s / 30;
			}
		break;
		case 0x14: log[log::verbose_debug] << "1080i50 100 Mbps";
			if (fields ==50) {
				log[log::verbose_debug] << "625/PAL 50Mbps";
				res = {1440, 1080};
				dur = 1_s / 25;
			} else {
				log[log::verbose_debug] << "1080i60 100 Mbps";
				res = {1280, 1080}; //??
				dur = 1_s / 30;
			}break;
		case 0x18: log[log::verbose_debug] << "DVCPRO HD 720p";
			if (fields ==50) {
				log[log::verbose_debug] << "720p50 100 Mbps";
				res = {960, 720};
				dur = 1_s / 25;
			} else {
				log[log::verbose_debug] << "720p60 100 Mbps";
				res = {960, 720};
				dur = 1_s / 30;
			}break;
		default: log[log::debug] << "Unknown signal type";
		return false;
	}
	log[log::verbose_debug] << "Setting DV " << res;
	frame->set_resolution(res);
	frame->set_duration(dur);
	frame->set_format(yuri::core::compressed_frame::dv);
	return true;
}


}

}


