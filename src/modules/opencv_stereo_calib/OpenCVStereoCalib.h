/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OpenCVStereoCalib.h
 * Author: user
 *
 * Created on 9. únor 2016, 15:18
 */

#ifndef OPENCVSTEREOCALIB_H
#define OPENCVSTEREOCALIB_H

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri{
namespace opencvstereocalib{
class OpenCVStereoCalib: public core::SpecializedIOFilter<core::RawVideoFrame>{
    using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    OpenCVStereoCalib(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    OpenCVStereoCalib(const OpenCVStereoCalib& orig);
    virtual ~OpenCVStereoCalib() noexcept;
private:
    virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
    virtual bool set_param(const core::Parameter& param) override;
};
}
}
#endif /* OPENCVSTEREOCALIB_H */

