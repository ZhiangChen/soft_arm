#ifndef SUBSCRIBER_
#define SUBSCRIBER_

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>

class PoseSubscriber
{
public:
    PoseSubscriber(ros::NodeHandle* nodehandle);
    void publish_poses(double timestep);

private:
    ros::NodeHandle _nh;
    ros::Subscriber _sub1;
    ros::Subscriber _sub2;
    ros::Subscriber _sub3;
    ros::Subscriber _sub4;
    ros::Publisher _pub;
    geometry_msgs::PoseArray _n_ps;

    bool _status1;
    bool _status2;
    bool _status3;
    bool _status4;
    void _callback1(const geometry_msgs::PoseStamped& ps);
    void _callback2(const geometry_msgs::PoseStamped& ps);
    void _callback3(const geometry_msgs::PoseStamped& ps);
    void _callback4(const geometry_msgs::PoseStamped& ps);

};

#endif
