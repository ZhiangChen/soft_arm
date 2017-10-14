#include <soft_arm/StateSubscriber.h>

StateSubscriber::StateSubscriber(ros::NodeHandle* nodehandle): _nh(*nodehandle)
{
    _sub1 = _nh.subscribe("Robot_1/pose", 1, &StateSubscriber::_callback1, this);
    _sub2 = _nh.subscribe("Robot_2/pose", 1, &StateSubscriber::_callback2, this);
    _sub3 = _nh.subscribe("Robot_3/pose", 1, &StateSubscriber::_callback3, this);
    _sub4 = _nh.subscribe("Robot_4/pose", 1, &StateSubscriber::_callback4, this);
    _sub5 = _nh.subscribe("Robot_5/pose", 1, &StateSubscriber::_callback5, this);
    _status1 = false;
    _status2 = false;
    _status3 = false;
    _status4 = false;
    _status5 = false;
    ROS_INFO("Initialized Subscribers");

    _pub = _nh.advertise<geometry_msgs::PoseArray>("robot_poses", 1, true);
    ROS_INFO("Initialized Publisher");

    _n_ps.poses.resize(5);
    _n_ps.header.frame_id = "raw_data";
}

void StateSubscriber::publish_state(double rate)
{
    while((!_status1 || !_status2 || !_status3 || !_status4 || !_status5) && ros::ok())
    {
        ros::spinOnce();
    }
    _status1 = false;
    _status2 = false;
    _status3 = false;
    _status4 = false;
    _status5 = false;
    ros::Rate(rate).sleep();
    ROS_INFO(".");
    _pub.publish(_n_ps);
}

void StateSubscriber::_callback1(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[0] = ps.pose;
    _status1 = true;
}

void StateSubscriber::_callback2(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[1] = ps.pose;
    _status2 = true;
}

void StateSubscriber::_callback3(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[2] = ps.pose;
    _status3 = true;
}

void StateSubscriber::_callback4(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[3] = ps.pose;
    _status4 = true;
}

void StateSubscriber::_callback5(const geometry_msgs::PoseStamped& ps)
{
    _n_ps.poses[4] = ps.pose;
    _status5 = true;
}
