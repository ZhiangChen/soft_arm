#include <ros/ros.h>
#include <soft_arm/PoseSubscriber.h>
#include <geometry_msgs/PoseStamped.h>

int main(int argc, char **argv)
{
    ros::init(argc,argv,"pose_subs");
    ros::NodeHandle n;
    PoseSubscriber ps;
}