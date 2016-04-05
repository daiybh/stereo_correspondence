/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SNCC.cpp
 * Author: user
 * 
 * Created on 4. dubna 2016, 10:52
 */

#include "CudaSNCC.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include <limits>
#include <c++/4.9/limits>

namespace yuri {
    namespace cuda_sncc {

        IOTHREAD_GENERATOR(CudaSNCC)
        MODULE_REGISTRATION_BEGIN("cuda_sncc")
        REGISTER_IOTHREAD("cuda_sncc", CudaSNCC)
        MODULE_REGISTRATION_END()
        CudaSNCC::CudaSNCC(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters) :
        core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 2, std::string("sncc")) {
            IOTHREAD_INIT(parameters)
                    //set_supported_formats({core::raw_format::rgba32});
            supported_formats_.push_back(core::raw_format::y8);
        }

        core::Parameters CudaSNCC::configure() {
            core::Parameters p = base_type::configure();
            p.set_description("SNCC Disparity computation");
            p["num_disparities"]["Number of disparities"]=16;
            return p;
        }


        

        std::vector<core::pFrame> CudaSNCC::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) {
            converter_left = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            converter_right = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            core::pRawVideoFrame left_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_left->convert_to_cheapest(std::get<0>(frames), supported_formats_));
            core::pRawVideoFrame right_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_right->convert_to_cheapest(std::get<1>(frames), supported_formats_));
            size_t w = left_frame->get_width();
            size_t h = left_frame->get_height();
            
            return {left_frame};
        }

        bool CudaSNCC::set_param(const core::Parameter& param) {
            if (assign_parameters(param)
                    (num_disparities, "num_disparities"))
                return true;
            return core::MultiIOFilter::set_param(param);
        }

        CudaSNCC::~CudaSNCC() noexcept {
        }
    }
}
