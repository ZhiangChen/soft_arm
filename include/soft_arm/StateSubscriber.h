#ifndef SUBSCRIBER_
#define SUBSCRIBER_

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <deque>

class StateSubscriber
{
public:
    StateSubscriber(ros::NodeHandle* nodehandle);
    void set_normalizer(double x, double y, double z, double s);
    void publish_state(double timestep);

private:
    ros::NodeHandle _nh;
    ros::Subscriber _sub1;
    ros::Subscriber _sub2;
    ros::Subscriber _sub3;
    ros::Subscriber _sub4;
    ros::Subscriber _sub5;
    ros::Publisher _pub;
    geometry_msgs::PoseArray _n_ps;

    double _x;
    double _y;
    double _z;
    double _s;

    bool _status1;
    bool _status2;
    bool _status3;
    bool _status4;
    bool _status5;
    std::deque<geometry_msgs::Pose> _pose1;
    std::deque<geometry_msgs::Pose> _pose2;
    std::deque<geometry_msgs::Pose> _pose3;
    std::deque<geometry_msgs::Pose> _pose4;
    std::deque<geometry_msgs::Pose> _pose5;
    void _callback1(const geometry_msgs::PoseStamped& ps);
    void _callback2(const geometry_msgs::PoseStamped& ps);
    void _callback3(const geometry_msgs::PoseStamped& ps);
    void _callback4(const geometry_msgs::PoseStamped& ps);
    void _callback5(const geometry_msgs::PoseStamped& ps);

    bool _is_valid(geometry_msgs::Pose p);
    geometry_msgs::Pose _average(std::deque<geometry_msgs::Pose> poses);

};

#endif
