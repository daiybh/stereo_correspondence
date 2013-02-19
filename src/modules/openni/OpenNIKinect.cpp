/*
 * OpenNIKinect.cpp
 *
 *  Created on: 19.2.2013
 *      Author: neneko
 */

#include "OpenNIKinect.h"
#include "yuri/config/RegisteredClass.h"
namespace yuri {
namespace OpenNIKinect {

REGISTER("openni_kinect",OpenNIKinect)

IO_THREAD_GENERATOR(OpenNIKinect)


shared_ptr<config::Parameters> OpenNIKinect::configure()
{
	shared_ptr<config::Parameters> p = BasicIOThread::configure();
	p->set_description("OpenNIKinect Kinect source.");
	(*p)["enable_depth"]["Enable depth sensors"]=true;
	(*p)["enable_ir"]	["Enable ir sensors"]	=false;
	(*p)["enable_color"]["Enable color sensors"]	=false;
	(*p)["enable_sync"]["Enable depth-color sync"]	=true;
	(*p)["enable_registration"]["Enable depth-color registration"]	=false;
	(*p)["sensors"]["Max number of activated sensors"]=1;

	p->set_max_pipes(0,1);
	return p;
}

OpenNIKinect::OpenNIKinect(log::Log &log_,io::pThreadBase parent, config::Parameters &parameters):
io::BasicIOThread(log_,parent,0,16,std::string("OpenNIKinect")),enable_depth(true),
enable_ir(false),enable_color(false),enable_sync(true),enable_registration(false),
max_sensors(1)
{
	latency=100;
	IO_THREAD_INIT("OpenNIKinect")
	//if (!dummy_name.empty()) log[info] << "Got name " << dummy_name <<"\n";
	if (openni::OpenNI::initialize()!=  openni::STATUS_OK) {
		log[log::fatal]<<"Failed to initialize OpenNI! Error: "<< openni::OpenNI::getExtendedError() <<"\n";
		throw yuri::exception::InitializationFailed("Failed to initialize OpenNI!");
	}
	openni::Array< openni::DeviceInfo > device_infos;
	openni::OpenNI::enumerateDevices(&device_infos);
	if (!device_infos.getSize()) {
		log[log::fatal]<<"No devices found!\n";
		throw yuri::exception::InitializationFailed("No devices found!");
	}
	log[log::debug] << "Enabled:  depth: " << enable_depth << ", ir: "<< enable_ir<< ", color: " <<enable_color<<"\n";
	yuri::size_t activated = 0;
	for (int i=0;i<device_infos.getSize();++i) {
		log[log::info] << "Device " << i <<": " << device_infos[i].getName() << ", uri: " << device_infos[i].getUri() << "\n";
		if (activated<max_sensors) {
			pDevice dev(new openni::Device());
			if (dev->open(device_infos[i].getUri())!=openni::STATUS_OK) {
				log[log::error] << "Failed to open device\n";
				continue;
			}
			log[log::debug] << "Device opened successfully\n";

//			if (dev->setImageRegistrationMode(enable_registration?openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR:openni::IMAGE_REGISTRATION_OFF)
//					!=openni::STATUS_OK) {
//				log[log::warning] << "Failed to " << (enable_registration?"enable":"disable") << " color-depth registration\n";
//			}
			if (dev->setDepthColorSyncEnabled(enable_sync)!=openni::STATUS_OK) {
				log[log::warning] << "Failed to " << (enable_sync?"enable":"disable") << " color-depth sync\n";
			}
			int found_streams=0;
			if (enable_depth) if (enable_sensor(dev,openni::SENSOR_DEPTH)) found_streams++;
			if (enable_ir) if (enable_sensor(dev,openni::SENSOR_IR)) found_streams++;
			if (enable_color) if (enable_sensor(dev,openni::SENSOR_COLOR)) found_streams++;
			if (found_streams>0) {
				log[log::debug] << "Storing device\n";
				devices.push_back(dev);
				activated++;
			} else {
				dev->close();
			}
		}

	}
	if (!activated) {
		throw exception::InitializationFailed("Failed to create at least 1 node");
	}


}
OpenNIKinect::~OpenNIKinect()
{
	openni::OpenNI::shutdown();
}

namespace {
std::string get_type_name(const openni::SensorType type) {
	switch(type) {
		case openni::SENSOR_DEPTH: return "depth";
		case openni::SENSOR_IR: return "ir";
		case openni::SENSOR_COLOR: return "color";
		default: return "unknowd";
	}
}
openni::VideoMode get_def_mode(const openni::SensorType type) {
	openni::VideoMode mode_;
	mode_.setResolution(640, 480);
	mode_.setFps(60);
	switch(type) {
		case openni::SENSOR_DEPTH: mode_.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_100_UM); break;
		case openni::SENSOR_IR: mode_.setPixelFormat(openni::PIXEL_FORMAT_GRAY16); break;
		case openni::SENSOR_COLOR: mode_.setPixelFormat(openni::PIXEL_FORMAT_RGB888); break;
		default: break;
	}
	return mode_;
}
}

bool OpenNIKinect::enable_sensor(pDevice dev, openni::SensorType type)
{
	openni::VideoMode new_mode;
	if (dev->hasSensor(type)) {
		log[log::debug] << get_type_name(type) << "Sensor found\n";
		const openni::SensorInfo* sinfo = dev->getSensorInfo(type);
		const openni::Array<openni::VideoMode>& modes = sinfo->getSupportedVideoModes();
		if (!modes.getSize()) {
			log[log::debug] << "Failed to get supported video modes\n";
			new_mode = get_def_mode(type);
		} else new_mode = modes[0];
		for (int m=0;m<modes.getSize();++m) {
			log[log::debug] << "Supports mode: " << modes[m].getResolutionX() <<
					"x" <<modes[m].getResolutionY()<<" at " << modes[m].getFps() << "fps\n";
		}
	} else {
		log[log::debug] << "No "<< get_type_name(type)<<" sensor!\n";
		return false;
	}

	pVideoStream dstream (new openni::VideoStream());
	if (dstream->create(*dev,type)!=openni::STATUS_OK) {
		log[log::error] << "Failed to create "<< get_type_name(type)<<"stream\n";
		return false;
	}
	if (dstream->setVideoMode(new_mode)!=openni::STATUS_OK) {
		log[log::warning] << "Failed to set video mode\n";
	}
	video_streams.push_back(dstream);
	return true;
}
void OpenNIKinect::run()
{
	IO_THREAD_PRE_RUN
	std::vector<openni::VideoStream*> stream_pointers;
	BOOST_FOREACH(pVideoStream str, video_streams) {
		if (str->start()!=openni::STATUS_OK) {
			log[log::warning] << "Failed to start stream\n";
		} else {
			stream_pointers.push_back(str.get());
		}
	}
	last_number.resize(stream_pointers.size(),-1);
	int index;
	while (still_running()) {

		if (openni::OpenNI::waitForAnyStream(&stream_pointers[0],static_cast<int>(stream_pointers.size()),
				&index,openni::TIMEOUT_NONE)==openni::STATUS_OK) {
			openni::VideoFrameRef frame_ref;
			assert(index>=0 && index < stream_pointers.size());
			if (stream_pointers[index]->readFrame(&frame_ref)!=openni::STATUS_OK) {
				log[log::warning] << "Failed to read frame from stream " << index << "\n";
			} else {
				const yuri::size_t width = frame_ref.getWidth();
				const yuri::size_t height = frame_ref.getHeight();
				const yuri::size_t num = frame_ref.getFrameIndex();
				log[log::debug] << "Frame "<< num <<" arrived at stream " << index << ", "
						<< width << "x" << height <<
						". stride: "<< frame_ref.getStrideInBytes() <<"\n";
				yuri::ssize_t missing = 0;
				if (last_number[index]>0) {
					missing = last_number[index] +1 - num;
					if (missing) {
						if (missing > 0) {
							log[log::warning] << "Missing " << missing <<" frames! Replicating\n";
						} else {
							log[log::error] << "Repeated frame!! This should NOT happen...\n";
							missing = 0;
						}
					}
				}
				io::pBasicFrame frame = io::BasicIOThread::allocate_frame_from_memory(
						reinterpret_cast<const yuri::ubyte_t*>(frame_ref.getData()),
						frame_ref.getDataSize(),true);
				openni::VideoMode mode = frame_ref.getVideoMode();
				yuri::format_t format = YURI_FMT_NONE;
				switch (mode.getPixelFormat()) {
				case openni::PIXEL_FORMAT_DEPTH_1_MM:format = YURI_FMT_DEPTH16;break;
				case openni::PIXEL_FORMAT_DEPTH_100_UM:format = YURI_FMT_DEPTH16;break;
				case openni::PIXEL_FORMAT_SHIFT_9_2:
				case openni::PIXEL_FORMAT_SHIFT_9_3:
				case openni::PIXEL_FORMAT_JPEG: log[log::debug]<< "Frame is in undocumented format. Frame size: " <<frame_ref.getDataSize() <<"\n";break;
				case openni::PIXEL_FORMAT_RGB888: format = YURI_FMT_RGB24;break;
				case openni::PIXEL_FORMAT_YUV422: format = YURI_FMT_YUV422;break;
				case openni::PIXEL_FORMAT_GRAY8: format = YURI_FMT_DEPTH16;break;
				case openni::PIXEL_FORMAT_GRAY16: format = YURI_FMT_DEPTH16;break;

				}
				if (format != YURI_FMT_NONE) {
					io::pFrameInfo fi(new io::FrameInfo());
					fi->max_value = stream_pointers[index]->getMaxPixelValue();
					fi->min_value = stream_pointers[index]->getMinPixelValue();
					frame->set_info(fi);
					if (missing) {
						for (ssize_t i=0;i<missing;++i) push_video_frame(index,frame,format,width,height);
					}
					push_video_frame(index,frame,format,width,height);
					last_number[index]=num;
				}

			}
			frame_ref.release();
		} else {
			sleep(latency);
		}
	}

	BOOST_FOREACH(pVideoStream str, video_streams) {
		str->stop();
		str->destroy();
	}
	BOOST_FOREACH(pDevice dev, devices) {
		dev->close();
	}
	IO_THREAD_POST_RUN
}
bool OpenNIKinect::set_param(config::Parameter& param)
{
	log[log::info] << "Got param " << param.name <<": " << param.get<int>() <<"\n";
	if (param.name == "enable_depth") {
		enable_depth = param.get<bool>();
	} else if (param.name == "enable_ir") {
		enable_ir = param.get<bool>();
	} else if (param.name == "enable_color") {
		enable_color = param.get<bool>();
	} else if (param.name == "enable_sync") {
		enable_sync = param.get<bool>();
	} else if (param.name == "enable_registration") {
		enable_registration = param.get<bool>();
	} else if (param.name == "sensors") {
		max_sensors = param.get<yuri::size_t>();
	} else return BasicIOThread::set_param(param);
	return true;
}


} /* namespace OpenNIKinect */
} /* namespace yuri */
