/*!
 * @file 		flycap_c_helpers.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		4. 6. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef SRC_MODULES_FLYCAP_FLYCAP_C_HELEPRS_H_
#define SRC_MODULES_FLYCAP_FLYCAP_C_HELEPRS_H_

#if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ > 7))
#define CAN_DISABLE_PEDANTIC
#endif

#ifdef CAN_DISABLE_PEDANTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include "C/FlyCapture2_C.h"

#ifdef CAN_DISABLE_PEDANTIC
#pragma GCC diagnostic pop
#undef CAN_DISABLE_PEDANTIC
#endif


#include "yuri/core/thread/ThreadBase.h"
//#include "yuri/core/utils/time_types.h"
#include "yuri/exception/InitializationFailed.h"
namespace yuri {
namespace flycap {



struct flycap_camera_t {
	enum class state_t {
		initialized,
		connected,
		started
	};

	using ctx_ref_t = std::add_lvalue_reference<fc2Context>::type;
	using ctx_ptr_t = std::add_pointer<fc2Context>::type;

	flycap_camera_t():ctx(nullptr),state(state_t::initialized)
	{
		fc2CreateContext(&ctx);
	}

	~flycap_camera_t() noexcept {
		stop();
		disconnect();
		fc2DestroyContext(ctx);
		core::ThreadBase::sleep(100_ms);
	}

	flycap(const flycap_camera_t&) = delete;
	flycap(flycap_camera_t&&) = delete;
	flycap_camera_t& operator=(const flycap_camera_t&) = delete;
	flycap_camera_t& operator=(flycap_camera_t&&) = delete;
	operator ctx_ref_t() {
		return ctx;
	}

	ctx_ptr_t operator&() {
		return &ctx;
	}

	void connect(unsigned int index, unsigned int serial = 0)
	{
		fc2PGRGuid guid;
		fc2Error error;
		if (!serial) {
			error = fc2GetCameraFromIndex( ctx, index, &guid );
			if (error != FC2_ERROR_OK)
			{
				throw exception::InitializationFailed("Failed to get camera with index " + std::to_string(index));
			}
		} else {
			error = fc2GetCameraFromSerialNumber( ctx, serial, &guid );
			if (error != FC2_ERROR_OK)
			{
				throw exception::InitializationFailed("Failed to get camera with serial " + std::to_string(serial));
			}
		}
		error = fc2Connect( ctx, &guid );
		if (error != FC2_ERROR_OK) {
			throw exception::InitializationFailed("Failed to connect to camera");
		}
		state = state_t::connected;
	}

	void disconnect() {
		stop();
		if (state == state_t::connected) {
			fc2Disconnect(ctx);
		}
		state = state_t::initialized;
	}

	void start()
	{
		if (state != state_t::connected) {
			throw exception::InitializationFailed("Camera not connected!");
		}
		if (fc2StartCapture(ctx)!= FC2_ERROR_OK) {
			throw exception::InitializationFailed("Failed to start capture");
		}

	}

	fc2CameraInfo get_camera_info()
	{
		fc2CameraInfo cam_info;
		if (state != state_t::connected) {
			throw exception::InitializationFailed("Camera not connected!");
		}
		if (fc2GetCameraInfo(ctx,  &cam_info ) != FC2_ERROR_OK)
		{
			throw exception::InitializationFailed("Failed to query camera info");
		}
		return cam_info;
	}

	void stop()
	{
		if (state == state_t::started) {
			fc2StopCapture(ctx);
		}
		state = state_t::connected;
	}
	fc2Context ctx;
	state_t state;
};


}
}




#endif /* SRC_MODULES_FLYCAP_FLYCAP_C_HELEPRS_H_ */

