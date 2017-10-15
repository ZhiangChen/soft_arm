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

    _pub = _nh.advertise<geometry_msgs::PoseArray>("agent_state", 1, true);
    ROS_INFO("Initialized Publisher");

    _n_ps.poses.resize(5);
    _n_ps.header.frame_id = "raw_data";

    _x = 0;
    _y = 0;
    _z = 0;
    _s = 1;
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

bool StateSubscriber::_is_valid(geometry_msgs::Pose p)
{
    return true;
}

void StateSubscriber::set_normalizer(double x, double y, double z, double s)
{
    _x = x;
    _y = y;
    _z = z;
    _s = s;
}

geometry_msgs::Pose StateSubscriber::_average(std::deque<geometry_msgs::Pose> poses)
{
    geometry_msgs::Pose pose1 = poses[0];
    geometry_msgs::Pose pose2 = poses[1];
    geometry_msgs::Pose pose3 = poses[2];
    geometry_msgs::Pose pose;


    pose.position.x = ((pose1.position.x + pose2.position.x + pose3.position.x)/3.0 - _x)*_s;
    pose.position.y = ((pose1.position.y + pose2.position.y + pose3.position.y)/3.0 - _y)*_s;
    pose.position.z = ((pose1.position.z + pose2.position.z + pose3.position.z)/3.0 - _z)*_s;
    pose.orientation.x = (pose1.orientation.x + pose2.orientation.x + pose3.orientation.x)/3.0;
    pose.orientation.y = (pose1.orientation.y + pose2.orientation.y + pose3.orientation.y)/3.0;
    pose.orientation.z = (pose1.orientation.z + pose2.orientation.z + pose3.orientation.z)/3.0;
    pose.orientation.w = (pose1.orientation.w + pose2.orientation.w + pose3.orientation.w)/3.0;
    return pose;
}


void StateSubscriber::_callback1(const geometry_msgs::PoseStamped& ps)
{
    if (_is_valid(ps.pose))
    {
        _pose1.push_back(ps.pose);
        if (_pose1.size()==4)
        {
            _pose1.pop_front();
            _n_ps.poses[0] = _average(_pose1);
            _status1 = true;
        }
    }

}

void StateSubscriber::_callback2(const geometry_msgs::PoseStamped& ps)
{
    if (_is_valid(ps.pose))
    {
        _pose2.push_back(ps.pose);
        if (_pose2.size()==4)
        {
            _pose2.pop_front();
            _n_ps.poses[1] = _average(_pose2);
            //ROS_INFO("%f", _n_ps.poses[1].position.x);
            _status2 = true;
        }
    }

}

void StateSubscriber::_callback3(const geometry_msgs::PoseStamped& ps)
{
    if (_is_valid(ps.pose))
    {
        _pose3.push_back(ps.pose);
        if (_pose3.size()==4)
        {
            _pose3.pop_front();
            _n_ps.poses[2] = _average(_pose3);
            _status3 = true;
        }
    }

}

void StateSubscriber::_callback4(const geometry_msgs::PoseStamped& ps)
{
    if (_is_valid(ps.pose))
    {
        _pose4.push_back(ps.pose);
        if (_pose4.size()==4)
        {
            _pose4.pop_front();
            _n_ps.poses[3] = _average(_pose4);
            _status4 = true;
        }
    }

}

void StateSubscriber::_callback5(const geometry_msgs::PoseStamped& ps)
{
    if (_is_valid(ps.pose))
    {
        _pose5.push_back(ps.pose);
        if (_pose5.size()==4)
        {
            _pose5.pop_front();
            _n_ps.poses[4] = _average(_pose5);
            _status5 = true;
        }
    }

}
