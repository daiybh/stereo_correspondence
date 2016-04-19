/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DimencoOut.cpp
 * Author: user
 * 
 * Created on 12. dubna 2016, 13:50
 */

#include <string.h>

#include "DimencoOut.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri {
    namespace dimencoout {

        IOTHREAD_GENERATOR(DimencoOut)

        MODULE_REGISTRATION_BEGIN("dimenco_out")
        REGISTER_IOTHREAD("dimenco_out", DimencoOut)
        MODULE_REGISTRATION_END()

        core::Parameters DimencoOut::configure() {
            core::Parameters p = base_type::configure();
            p.set_description("Output to Dimenco display");
            return p;
        }

        DimencoOut::~DimencoOut() {

        }

        DimencoOut::DimencoOut(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters) :
        base_type(log_, parent, std::string("dimenco_out")) {
            IOTHREAD_INIT(parameters)
            set_supported_formats({core::raw_format::rgb24});
        }

        core::pFrame DimencoOut::do_special_single_step(core::pRawVideoFrame frame) {
            int height = frame->get_height();
            int width = frame->get_width();
            
                if(width != 3840 || (height != 1080 && height != 2160)){
                    log[log::info]<<"Not adding service bits";
                    return frame;
                }
            core::pRawVideoFrame output = core::RawVideoFrame::create_empty(core::raw_format::rgb24,frame->get_resolution());
            uint8_t *frame_data=PLANE_RAW_DATA(frame,0);
            uint8_t *out_data=PLANE_RAW_DATA(output,0);
            for(int i=0;i<height;i++){
                if(i % 2 == 0){
                    memcpy(&out_data[i*width*3],&frame_data[i*width*3],sizeof(uint8_t)*width*3);
                }
            }
            set_service_bits(PLANE_RAW_DATA(output,0));
            return output;
        }

        void DimencoOut::set_service_bits(uint8_t* frame_data) {
            frame_data[0 * 3] = 16;
            frame_data[0 * 3 + 1] = 16;
            frame_data[0 * 3 + 2] = 255;
            frame_data[2 * 3] = 16;
            frame_data[2 * 3 + 1] = 16;
            frame_data[2 * 3 + 2] = 255;
            frame_data[4 * 3] = 16;
            frame_data[4 * 3 + 1] = 16;
            frame_data[4 * 3 + 2] = 255;
            frame_data[6 * 3] = 16;
            frame_data[6 * 3 + 1] = 16;
            frame_data[6 * 3 + 2] = 255;
            frame_data[14 * 3] = 16;
            frame_data[14 * 3 + 1] = 16;
            frame_data[14 * 3 + 2] = 255;
            frame_data[30 * 3] = 16;
            frame_data[30 * 3 + 1] = 16;
            frame_data[30 * 3 + 2] = 255;
            frame_data[96 * 3] = 16;
            frame_data[96 * 3 + 1] = 16;
            frame_data[96 * 3 + 2] = 255;
            frame_data[98 * 3] = 16;
            frame_data[98 * 3 + 1] = 16;
            frame_data[98 * 3 + 2] = 255;
            frame_data[100 * 3] = 16;
            frame_data[100 * 3 + 1] = 16;
            frame_data[100 * 3 + 2] = 255;
            frame_data[102 * 3] = 16;
            frame_data[102 * 3 + 1] = 16;
            frame_data[102 * 3 + 2] = 255;
            frame_data[104 * 3] = 16;
            frame_data[104 * 3 + 1] = 16;
            frame_data[104 * 3 + 2] = 255;
            frame_data[110 * 3] = 16;
            frame_data[110 * 3 + 1] = 16;
            frame_data[110 * 3 + 2] = 255;
            frame_data[114 * 3] = 16;
            frame_data[114 * 3 + 1] = 16;
            frame_data[114 * 3 + 2] = 255;
            frame_data[116 * 3] = 16;
            frame_data[116 * 3 + 1] = 16;
            frame_data[116 * 3 + 2] = 255;
            frame_data[120 * 3] = 16;
            frame_data[120 * 3 + 1] = 16;
            frame_data[120 * 3 + 2] = 255;
            frame_data[146 * 3] = 16;
            frame_data[146 * 3 + 1] = 16;
            frame_data[146 * 3 + 2] = 255;
            frame_data[148 * 3] = 16;
            frame_data[148 * 3 + 1] = 16;
            frame_data[148 * 3 + 2] = 255;
            frame_data[150 * 3] = 16;
            frame_data[150 * 3 + 1] = 16;
            frame_data[150 * 3 + 2] = 255;
            frame_data[152 * 3] = 16;
            frame_data[152 * 3 + 1] = 16;
            frame_data[152 * 3 + 2] = 255;
            frame_data[156 * 3] = 16;
            frame_data[156 * 3 + 1] = 16;
            frame_data[156 * 3 + 2] = 255;
            frame_data[160 * 3] = 16;
            frame_data[160 * 3 + 1] = 16;
            frame_data[160 * 3 + 2] = 255;
            frame_data[162 * 3] = 16;
            frame_data[162 * 3 + 1] = 16;
            frame_data[162 * 3 + 2] = 255;
            frame_data[164 * 3] = 16;
            frame_data[164 * 3 + 1] = 16;
            frame_data[164 * 3 + 2] = 255;
            frame_data[166 * 3] = 16;
            frame_data[166 * 3 + 1] = 16;
            frame_data[166 * 3 + 2] = 255;
            frame_data[172 * 3] = 16;
            frame_data[172 * 3 + 1] = 16;
            frame_data[172 * 3 + 2] = 255;
            frame_data[182 * 3] = 16;
            frame_data[182 * 3 + 1] = 16;
            frame_data[182 * 3 + 2] = 255;
            frame_data[186 * 3] = 16;
            frame_data[186 * 3 + 1] = 16;
            frame_data[186 * 3 + 2] = 255;
            frame_data[242 * 3] = 16;
            frame_data[242 * 3 + 1] = 16;
            frame_data[242 * 3 + 2] = 255;
            frame_data[244 * 3] = 16;
            frame_data[244 * 3 + 1] = 16;
            frame_data[244 * 3 + 2] = 255;
            frame_data[246 * 3] = 16;
            frame_data[246 * 3 + 1] = 16;
            frame_data[246 * 3 + 2] = 255;
            frame_data[248 * 3] = 16;
            frame_data[248 * 3 + 1] = 16;
            frame_data[248 * 3 + 2] = 255;
            frame_data[266 * 3] = 16;
            frame_data[266 * 3 + 1] = 16;
            frame_data[266 * 3 + 2] = 255;
            frame_data[276 * 3] = 16;
            frame_data[276 * 3 + 1] = 16;
            frame_data[276 * 3 + 2] = 255;
            frame_data[278 * 3] = 16;
            frame_data[278 * 3 + 1] = 16;
            frame_data[278 * 3 + 2] = 255;
            frame_data[280 * 3] = 16;
            frame_data[280 * 3 + 1] = 16;
            frame_data[280 * 3 + 2] = 255;
            frame_data[450 * 3] = 16;
            frame_data[450 * 3 + 1] = 16;
            frame_data[450 * 3 + 2] = 255;
            frame_data[456 * 3] = 16;
            frame_data[456 * 3 + 1] = 16;
            frame_data[456 * 3 + 2] = 255;
            frame_data[458 * 3] = 16;
            frame_data[458 * 3 + 1] = 16;
            frame_data[458 * 3 + 2] = 255;
            frame_data[460 * 3] = 16;
            frame_data[460 * 3 + 1] = 16;
            frame_data[460 * 3 + 2] = 255;
            frame_data[462 * 3] = 16;
            frame_data[462 * 3 + 1] = 16;
            frame_data[462 * 3 + 2] = 255;
            frame_data[466 * 3] = 16;
            frame_data[466 * 3 + 1] = 16;
            frame_data[466 * 3 + 2] = 255;
            frame_data[468 * 3] = 16;
            frame_data[468 * 3 + 1] = 16;
            frame_data[468 * 3 + 2] = 255;
            frame_data[470 * 3] = 16;
            frame_data[470 * 3 + 1] = 16;
            frame_data[470 * 3 + 2] = 255;
            frame_data[478 * 3] = 16;
            frame_data[478 * 3 + 1] = 16;
            frame_data[478 * 3 + 2] = 255;
            frame_data[480 * 3] = 16;
            frame_data[480 * 3 + 1] = 16;
            frame_data[480 * 3 + 2] = 255;
            frame_data[484 * 3] = 16;
            frame_data[484 * 3 + 1] = 16;
            frame_data[484 * 3 + 2] = 255;
            frame_data[486 * 3] = 16;
            frame_data[486 * 3 + 1] = 16;
            frame_data[486 * 3 + 2] = 255;
            frame_data[488 * 3] = 16;
            frame_data[488 * 3 + 1] = 16;
            frame_data[488 * 3 + 2] = 255;
            frame_data[490 * 3] = 16;
            frame_data[490 * 3 + 1] = 16;
            frame_data[490 * 3 + 2] = 255;
            frame_data[492 * 3] = 16;
            frame_data[492 * 3 + 1] = 16;
            frame_data[492 * 3 + 2] = 255;
            frame_data[498 * 3] = 16;
            frame_data[498 * 3 + 1] = 16;
            frame_data[498 * 3 + 2] = 255;
            frame_data[500 * 3] = 16;
            frame_data[500 * 3 + 1] = 16;
            frame_data[500 * 3 + 2] = 255;
            frame_data[504 * 3] = 16;
            frame_data[504 * 3 + 1] = 16;
            frame_data[504 * 3 + 2] = 255;
            frame_data[506 * 3] = 16;
            frame_data[506 * 3 + 1] = 16;
            frame_data[506 * 3 + 2] = 255;
            frame_data[508 * 3] = 16;
            frame_data[508 * 3 + 1] = 16;
            frame_data[508 * 3 + 2] = 255;
        }

        bool DimencoOut::set_param(const core::Parameter& param) {
            if (assign_parameters(param))
                return true;
            return base_type::set_param(param);
        }

    }
}
