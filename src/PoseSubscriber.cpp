#include <soft_arm/PoseSubscriber.h>

PoseSubscriber::PoseSubscriber(ros::NodeHandle* nodehandle): _nh(*nodehandle)
{
    _sub1 = _nh.subscribe("Robot_1/pose", 1, &PoseSubscriber::_callback1, this);
    _sub2 = _nh.subscribe("Robot_2/pose", 1, &PoseSubscriber::_callback2, this);
    _sub3 = _nh.subscribe("Robot_3/pose", 1, &PoseSubscriber::_callback3, this);
    _sub4 = _nh.subscribe("Robot_4/pose", 1, &PoseSubscriber::_callback4, this);
    _status1 = false;
    _status2 = false;
    _status3 = false;
    _status4 = false;
    ROS_INFO("Initialized Subscribers");

    _pub = _nh.advertise<geometry_msgs::PoseArray>("robot_poses", 1, true);
    ROS_INFO("Initialized Publisher");

    _n_ps.poses.resize(4);
    _n_ps.header.frame_id = "raw_data";
}

void PoseSubscriber::publish_poses(double rate)
{
    while((!_status1 || !_status2 || !_status3 || !_status4) && ros::ok())
    {
        ros::spinOnce();
    }
    _status1 = false;
    _status2 = false;
    _status3 = false;
    _status4 = false;
    ros::Rate(rate).sleep();
    std::cout<<'.';
    _pub.publish(_n_ps);
}

void PoseSubscriber::_callback1(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[0] = ps.pose;
    _status1 = true;
}

void PoseSubscriber::_callback2(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[1] = ps.pose;
    _status2 = true;
}

void PoseSubscriber::_callback3(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[2] = ps.pose;
    _status3 = true;
}

void PoseSubscriber::_callback4(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[3] = ps.pose;
    _status4 = true;
}