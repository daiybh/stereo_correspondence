/*
 * DVBSource.cpp
 *
 *  Created on: 20.2.2011
 *      Author: neneko
 */

#include "DVBSource.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include <linux/dvb/frontend.h>
#include <linux/dvb/dmx.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <unistd.h>

namespace yuri {

namespace dvb {



MODULE_REGISTRATION_BEGIN("dvb")
	REGISTER_IOTHREAD("dvbsource",DVBSource)
MODULE_REGISTRATION_END()


IOTHREAD_GENERATOR(DVBSource)

core::Parameters DVBSource::configure()
{
	core::Parameters p = core::IOThread::configure();
	p["adapter"]["Path to adapter to use"]="/dev/dvb/adapter0";
	p["frontend"]["Frontend to use"]="frontend0";
	p["frequency"]["Frequency to tune in Hz"]=730000000;
	p["pid"]["Channel to tune"]=273;
	p["apid"]["Audio channel to tune"]=-1;
//	p["format"]["Format of output. TS or None"]="NONE";
	p["chunk"]["Size of data chunk to read and output at time"]=1316;
	p["ts"]["Output TS"]=false;
	return p;
}

DVBSource::DVBSource(log::Log &log_, core::pwThreadBase parent, const core::Parameters& parameters)
	:core::IOThread(log_,parent,0,1,"DVB"),adapter_path_("/dev/dvb/adapter0"),
	 frontend_("frontend0"),frequency_(730000000),pid_(273),apid_(-1),
	 chunk_size_(1316),
	 fe_fd(-1),
	 demux_fd(-1),dvr_fd(-1),output_ts_(true)
{
	IOTHREAD_INIT(parameters)
	if (!init()) {
		throw exception::InitializationFailed("Failed to initialize dvb");
	}
}

DVBSource::~DVBSource() noexcept
{

}


void DVBSource::run()
{

	print_id();
//	size_t chunk = chunk_size_;//4096;//params["chunk"].get<size_t>();
//	yuri::ubyte_t *data = new yuri::ubyte_t[chunk];
	std::vector<uint8_t> data(chunk_size_, 0);
	int fd;
	yuri::format_t fmt;
	if (output_ts_) {
		fd = dvr_fd;
		fmt = core::compressed_frame::mpeg2ts;
	} else {
		fd = demux_fd;
		fmt = core::compressed_frame::mpeg2;
	}
	while (still_running()) {
		//if (not step()) request_end();
		int ret = read(fd,data.data(),chunk_size_);

		if (ret <= 0) {
			log[log::error] << "Failed to read (" << errno << "), fd: " << fd;
			request_end();
			break;
		}
		core::pCompressedVideoFrame frame_out = core::CompressedVideoFrame::create_empty(fmt, resolution_t{0,0}, data.data(), ret);
		push_frame(0, frame_out);

		//sleep(latency);
	}
	close_pipes();
	request_end();
}

bool DVBSource::step()
{
	return true;
}


bool DVBSource::init()
{
	std::string fedev = adapter_path_ + "/" + frontend_;
	std::string demuxdev = adapter_path_ + "/demux0";
	std::string dvrdev = adapter_path_ + "/dvr0";
	log[log::info] << "Opening frontend " << fedev;
	fe_fd = open(fedev.c_str(),O_RDWR);
	if (fe_fd < 0) {
		log[log::fatal] << "Failed to open" << fedev;
		return false;
	}


	log[log::debug] << "Querying frontend info ... ";
	dvb_frontend_info feinfo;
	int ret = ioctl(fe_fd,FE_GET_INFO,&feinfo);
	if (ret < 0) {
		log[log::fatal] << "Failed";;
		return false;
	}

	log[log::debug] << "Frontend info:";
	log[log::info] << "\tName:\t" << feinfo.name;
	//print_type(feinfo.type);
	//print_caps(feinfo.caps);
	log[log::debug] << "Freqs:\t " << feinfo.frequency_min << " - " << feinfo.frequency_max;


	log[log::debug] << "Setting frontend params ... ";
	dvb_frontend_parameters _params;
	_params.frequency= frequency_;
	ret = ioctl(fe_fd,FE_SET_FRONTEND,&_params);
	if (ret < 0) {
		log[log::fatal] << "Failed to set frequency";
		return false;
	}


	fe_status_t festat;
	log[log::debug] << "Querying frontend status ... ";
	ret = ioctl(fe_fd,FE_READ_STATUS,&festat);
	if (ret < 0) {
		log[log::warning] << "Failed to query frontend status";
	} else log[log::debug] << "Status: " << festat;

	log[log::debug] << "Querying signal strength ... ";
	int16_t sigstr;
	ret = ioctl(fe_fd,FE_READ_SIGNAL_STRENGTH,&sigstr);
	if (ret < 0) {
		log[log::warning] << "Failed to query singnal strength";
	}

	log[log::debug] << "Signal strength: " << sigstr;



	log[log::info] << "Opening demux" << demuxdev << " ... ";
	demux_fd = open(demuxdev.c_str(),O_RDWR);
	if (demux_fd < 0) {
		log[log::fatal] << "Failed to open" << demuxdev;
		return false;
	}


	dmx_pes_filter_params dmxpes;
	dmxpes.pid = pid_;
	dmxpes.input = DMX_IN_FRONTEND;
	if (output_ts_) dmxpes.output = DMX_OUT_TS_TAP;
	else dmxpes.output = DMX_OUT_TAP;
	dmxpes.pes_type = DMX_PES_VIDEO;
	log[log::debug] << "Setting filter ... ";
	ret = ioctl(demux_fd,DMX_SET_PES_FILTER,&dmxpes);
	if (ret < 0) {
		log[log::fatal] << "Failed";
		return false;
	}

	log[log::debug] << "Starting demux ... ";
	ret = ioctl(demux_fd,DMX_START);
	if (ret < 0) {
		log[log::fatal] << "Failed to start demux";
		return false;

	}
	if (output_ts_) {
		if (apid_>=0) {
			log[log::info] << "Opening demux" << demuxdev << " for audio ... ";\
				ademux_fd = open(demuxdev.c_str(),O_RDWR);
				if (ademux_fd < 0) {
					log[log::fatal] << "Failed to open" << demuxdev;
					return false;
				}


				dmx_pes_filter_params dmxpes;
				dmxpes.pid = apid_;
				dmxpes.input = DMX_IN_FRONTEND;
				/*if (output_ts_) */dmxpes.output = DMX_OUT_TS_TAP;
//				else dmxpes.output = DMX_OUT_TAP;
				dmxpes.pes_type = DMX_PES_AUDIO;
				log[log::debug] << "Setting filter ... ";
				ret = ioctl(ademux_fd,DMX_SET_PES_FILTER,&dmxpes);
				if (ret < 0) {
					log[log::fatal] << "Failed";
					return false;
				}

				log[log::debug] << "Starting demux ... ";
				ret = ioctl(ademux_fd,DMX_START);
				if (ret < 0) {
					log[log::fatal] << "Failed to start demux";
					return false;

				}
		}
		log[log::debug] << "Opening dvr " << dvrdev << " ... ";
		dvr_fd = open(dvrdev.c_str(),O_RDONLY);
		if (dvr_fd < 0) {
			log[log::fatal] << "Failed to open" << dvrdev;
			return false;
		}
	}
	return true;
}

bool DVBSource::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(adapter_path_, "adapter")
			(frontend_, "frontend")
			(frequency_, "frequency")
			(pid_, "pid")
			(apid_, "apid")
			(chunk_size_, "chunk_size")
			(output_ts_, "ts"))
		return true;
	return core::IOThread::set_param(param);
}


}

}

