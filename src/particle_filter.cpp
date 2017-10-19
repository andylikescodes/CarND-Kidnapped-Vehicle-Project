/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	// initialze useful variables
	//default_random_engine gen;
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// number of particles
	num_particles = 100;

	// initalize the x, y, theta from noise of a normal distribution
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_x);
	normal_distribution<double> dist_theta(theta, std_theta);

	// create particles
	for (int i = 0; i < num_particles; ++i){
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle p;
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1.0;

		weights.push_back(p.weight);
		particles.push_back(p);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	double std_x, std_y, std_yaw;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_yaw = std_pos[2];

	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_x);
	normal_distribution<double> dist_yaw(0, std_yaw);

	// predict particles for the next time stamp, considering the observation variances.
	for (int i = 0; i < num_particles; ++i){

		double noise_x, noise_y, noise_yaw;
		noise_x = dist_x(gen);
		noise_y = dist_y(gen);
		noise_yaw = dist_yaw(gen);

		Particle p = particles[i];

		//avoid dividing by zero.
		if (fabs(yaw_rate) > 0.0001){
			p.x = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + noise_x;
			p.y = p.y + velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + noise_y;
			
		}
		else {
			p.x = p.x + velocity*delta_t*cos(p.theta);
			p.y = p.y + velocity*delta_t*sin(p.theta);

		}

		p.theta = p.theta + yaw_rate * delta_t + noise_yaw;
		particles[i] = p;
	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	/* 
		for obs in observations:
			for predicted 
	*/

	// initialize a temporary minimum distance
	double min_dist = INFINITY;
	int min_id = -1;

	for (int i = 0; i < observations.size(); i ++){
		LandmarkObs o = observations[i];
		for (int j = 0; j < predicted.size(); j++){
			LandmarkObs p = predicted[j];
			double distance = dist(o.x, o.y, p.x, p.y);
			if (distance < min_dist){
				// cout << "observations" << endl;
				// cout << o.x << "   " << o.y << endl;
				// cout << "predicted" << endl;
				// cout << p.x << "   " << p.y << endl;
				// cout << "predicted id: " << p.id << endl;
				min_dist = distance;
				min_id = p.id;
			}
		}
		observations[i].id = min_id;
	}


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	// Interate over every particle
	for (int i = 0; i < particles.size(); i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		double weight = 1.0;

		// for each observation, transform from car coordinate to map coordinate.
		for (int j = 0; j < observations.size(); j++){
			LandmarkObs obs = observations[j];
			double obs_x = obs.x;
			double obs_y = obs.y;

			double pred_x = obs_x * cos(p_theta) - obs_y * sin(p_theta) + p_x;
			double pred_y = obs_x * sin(p_theta) + obs_y * cos(p_theta) + p_y;

			// Iterate over all the landmarks and find the shortest distance landmark from this observation
			double shortest_dist = sensor_range;
			double land_x;
			double land_y;


			for (int k = 0; k < map_landmarks.landmark_list.size(); k++){
				double temp_x = map_landmarks.landmark_list[k].x_f;
				double temp_y = map_landmarks.landmark_list[k].y_f;
				double distance = dist(pred_x, pred_y, temp_x, temp_y);

				if ( distance < shortest_dist){
					cout << "show me the distance: " << distance << endl;
					// Associate the shortest distance landmark with the observation.
					shortest_dist = distance;
					land_x = temp_x;
					land_y = temp_y;
				}
			}

			// calculate the weight using the associated landmark and the observation.
			double x_diff = pred_x - land_x;
			double y_diff = pred_y - land_y;
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double w = exp(-0.5*((x_diff * x_diff)/(std_x * std_x) + (y_diff * y_diff)/(std_y * std_y))) / (2.0*M_PI* std_x * std_y);
			cout << "show me the pred_x: " << pred_x << endl;
			cout << "show me the pred_y: " << pred_y << endl;
			cout << "show me the land_x: " << land_x << endl;
			cout << "show me the land_y: " << land_y << endl;
			cout << "show me the weight: " << w << endl;
			weight *= w;

		}

		// update the new weights to each particle.
		particles[i].weight = weight;
		weights[i] = weight;
	}



}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	// generate the distribution based on the weights array.
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resample_particles;

	// resample particles based on the weight for each particle.
	for (int i = 0; i < num_particles; i++)
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}

	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
