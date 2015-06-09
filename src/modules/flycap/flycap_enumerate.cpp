/*!
 * @file 		flycap_enumerate.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		5. 6. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#include "FlyCap.h"
#include "flycap_defs.h"
#include "yuri/core/utils/irange.h"
namespace yuri {
namespace flycap {

std::vector<core::InputDeviceInfo> FlyCap::enumerate()
{
	std::vector<core::InputDeviceInfo>  devices;
	std::vector<std::string> main_param_order = {"serial", "custom", "resolution", "format", "fps"};
	flycap_camera_t camera;
	unsigned int numCameras;
	fc2GetNumOfCameras(camera, &numCameras);


	for(const auto& idx: irange(numCameras)) {
		try {
			unsigned int serial = 0;
			fc2GetCameraSerialNumberFromIndex(camera, idx, &serial);
			camera.disconnect();
			camera.connect(idx, serial);

			core::InputDeviceInfo device;
			device.main_param_order = main_param_order;
			auto cam_info = camera.get_camera_info();

			device.device_name = cam_info.modelName;
			core::InputDeviceConfig cfg_base;
			cfg_base.params["serial"]=serial;

			for (const auto mode1: video_modes) {
				const auto res = mode1.first;
				for (const auto mode2: mode1.second) {
					const auto format = mode2.first;
					const auto fly_mode = mode2.second;

					for (const auto fpsi: frame_rates) {
						const auto fps = fpsi.first;
						const auto fly_fps = fpsi.second;
						BOOL supported = false;
						if (fc2GetVideoModeAndFrameRateInfo(camera, fly_mode, fly_fps, &supported) == FC2_ERROR_OK) {
							if (supported) {
								core::InputDeviceConfig cfg = cfg_base;
								cfg.params["custom"]=-1;
								cfg.params["resolution"]=res;
								cfg.params["fps"]=fps;
								const auto& fi = core::raw_format::get_format_info(format);
								if (fi.short_names.size() > 0) {
									cfg.params["format"] = fi.short_names[0];
								}
								device.configurations.push_back(cfg);
							}
						}
					}
				}
			}
			for (auto mode_idx: irange(static_cast<unsigned int>(FC2_NUM_MODES))) {
				fc2Format7Info f7info;
				BOOL supported = false;;
				f7info.mode = static_cast<fc2Mode>(mode_idx);
				fc2GetFormat7Info(camera, &f7info, &supported);
				if (supported) {
					resolution_t res {f7info.maxWidth, f7info.maxHeight};

					core::InputDeviceConfig cfg = cfg_base;
					cfg.params["custom"]=mode_idx;
					cfg.params["resolution"]=res;

					for (const auto& fmt: flycap_formats) {
						const auto& ffmt = fmt.first;
						if (f7info.pixelFormatBitField & ffmt) {
							core::InputDeviceConfig cfg2 = cfg;
							const auto& fi = core::raw_format::get_format_info(fmt.second);
							if (fi.short_names.size() > 0) {
								cfg2.params["format"] = fi.short_names[0];
							}
							device.configurations.push_back(cfg2);
						}
					}
				}
			}

			devices.push_back(std::move(device));
		}
		catch (std::exception& e) {
		}
	}
	return devices;
}



}
}



