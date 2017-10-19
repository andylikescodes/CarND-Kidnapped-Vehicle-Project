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

	float p_x, p_y, p_theta, l_x, l_y;
	int l_id;

	// for each particle
	for (int i = 0; i < particles.size(); i++){
		p_x = particles[i].x;
		p_y = particles[i].y;
		p_theta = particles[i].theta;

		vector<LandmarkObs> predicted; 
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
			l_id = map_landmarks.landmark_list[j].id_i;
			l_x = map_landmarks.landmark_list[j].x_f;
			l_y = map_landmarks.landmark_list[j].y_f;

			if (dist(p_x, p_y, l_x, l_y) <= sensor_range){
				LandmarkObs pred;
				pred.x = l_x;
				pred.y = l_y;
				pred.id = l_id;
				// cout << "Predicted landmark x: " << l_x << endl;
				// cout << "Predicted landmark y: " << l_y << endl;
				// cout << "Predicted landmark id: " << l_id << endl;
				predicted.push_back(pred);
			}
		}

		std::vector<LandmarkObs> transformed_obs;
		for (int j = 0; j < observations.size(); j ++){
			double x_t = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
			double y_t = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
			LandmarkObs obs;
			obs.id = observations[j].id;
			obs.x = x_t;
			obs.y = y_t;
			transformed_obs.push_back(obs);
		}

		dataAssociation(predicted, transformed_obs);

		// initialize weight
		long double weight = 1.0;

		for (int j = 0; j < transformed_obs.size(); j++){
			double trans_id = transformed_obs[j].id;
			LandmarkObs pred_landmark;
			for (int k = 0; k < predicted.size(); k++){
				if (predicted[k].id == trans_id){
					pred_landmark = predicted[k];
				}
			}

			double trans_x, trans_y, pred_x, pred_y, std_x, std_y;
			long double w;
			trans_x = transformed_obs[j].x;
			trans_y = transformed_obs[j].y;
			pred_x = pred_landmark.x;
			pred_y = pred_landmark.y;
			std_x = std_landmark[0];
			std_y = std_landmark[1];
			cout << "trans_x: " << trans_x << endl;
			cout << "trans_y: " << trans_y << endl;
			cout << "pred_x: " << pred_x << endl;
			cout << "pred_y: " << pred_y << endl;
			double x_diff = pred_x - trans_x;
			double y_diff = pred_y - trans_y;
			double nom = exp(-((x_diff*x_diff)/(2.0*std_x*std_x) + ((y_diff*y_diff)/(2.0*std_y*std_y))));
			double denom = 2.0*M_PI*std_x*std_y;
			// cout << "nom: " << nom << endl;
			// cout << "denom: " << denom << endl; 
			w = nom/denom;
			weight *= w;
			// cout << "little weights: " << w << endl;


		}
		// cout << "weight: " << weight << endl;  

		particles[i].weight = weight;
		weights[i] = weight;

	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resample_particles;

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
