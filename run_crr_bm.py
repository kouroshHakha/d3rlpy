import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pendulum-random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--advantage_type', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--weight_type', type=str, default='exp', choices=['binary', 'exp'])
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    crr = d3rlpy.algos.CRR(actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           batch_size=256,
                           weight_type=args.weight_type,
                           advantage_type=args.advantage_type,
                           target_update_type="soft",
                           use_gpu=args.gpu)

    crr.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CRR_{args.dataset}_{args.seed}_advtype_{args.advantage_type}_wtype_{args.weight_type}",
            tensorboard_dir='runs')


if __name__ == '__main__':
    main()
