/*!
 * @file 		TwopcProtocol.h
 * @author 		Anastasia Kuznetsova <kuzneana@gmail.com>
 * @date 		26. 3. 2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef TWOPCPROTOCOL_H_
#define TWOPCPROTOCOL_H_


#include "yuri/core/Module.h"
#include <memory>
#include <unordered_map>
#include <cmath>
#define TOLERANCE 0.000001

namespace yuri{
namespace synchronization {


inline event::pBasicEvent prepare_event(const uint64_t& id_sender, const index_t& data, const uint64_t& id_receiver=0, const bool& is_cohort=false){
    std::vector<event::pBasicEvent> vec;
    if(is_cohort)vec.push_back(std::make_shared<event::EventInt>(id_receiver));
    vec.push_back(std::make_shared<event::EventInt>(id_sender));
    vec.push_back(std::make_shared<event::EventInt>(data));
    return std::make_shared<event::EventVector>(std::move(vec));
}


 template <typename key_t, typename val_t>
 inline int control_cohorts_state(std::unordered_map<key_t, val_t>& map,val_t exp_val, val_t max_diff=7){
    if(map.size()!=0){
    for (const auto& kv : map) {
        if(exp_val - kv.second > max_diff){
            map.erase(kv.first);
            break;
        }
    }
    }
    return map.size();
}

 inline double calculate_fps(const core::pFrame frame){
     if(frame->get_duration().value==0) return 0.0;
     double fps = (1_s).value/frame->get_duration().value;
     return fps;
 }


inline duration_t set_timeout(const float& fps, const duration_t& curr_timeout, log::Log& log, const float& percent_=0.9){
     if(fps > TOLERANCE && curr_timeout >= 1_s/fps){
             auto calc_timeout = 1_s/fps *percent_;
             log[log::info]<<"The timeout is longer than expected";
             log[log::info]<<"Set timeout "<<calc_timeout;
             return calc_timeout;
     }
     return curr_timeout;
 }

 inline int calculate_percentage(const int& val,const int& pct){
     int temp_pct=pct;
     if(pct>100) {
         temp_pct=100;
     } else if(pct<0) temp_pct=0;
     return round(val*(temp_pct/100.0));
 }


 template <typename T>
 inline bool verify_coordinator_response(const event::EventVector& response,const T& id_coordinator){
     if(response.size()<2) return false;
     T id_sndr = event::lex_cast_value<T>(response[0]);
     if(id_coordinator != 0  && id_sndr!=id_coordinator) return false;
     return true;
 }


 template <typename T>
 inline bool verify_cohort_response(const event::EventVector& response, const T& id_coordinator){
     if(response.size()<3) return false;
     T id_receiver = event::lex_cast_value<T>(response[0]);
     if(id_receiver!=0 && id_receiver!=id_coordinator) return false;
     return true;
 }




























}
}

#endif /* TWOPCPROTOCOL_H_ */
