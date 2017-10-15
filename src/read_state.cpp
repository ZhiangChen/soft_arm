#include <ros/ros.h>
#include <soft_arm/StateSubscriber.h>
#include <geometry_msgs/PoseStamped.h>

int main(int argc, char **argv)
{
    ros::init(argc,argv,"read_state");
    ros::NodeHandle nh;
    StateSubscriber ss(&nh);
    while (ros::ok())
    {
        ss.publish_state(25);
    }
}
