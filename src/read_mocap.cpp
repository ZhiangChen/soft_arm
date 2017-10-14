#include <ros/ros.h>
#include <soft_arm/PoseSubscriber.h>
#include <geometry_msgs/PoseStamped.h>

int main(int argc, char **argv)
{
    ros::init(argc,argv,"read_mocap");
    ros::NodeHandle nh;
    PoseSubscriber ps(&nh);
    while (ros::ok())
    {
        ps.publish_poses(10);
    }
}