from demos.gmm8.problem import *


if __name__ == '__main__':

    DATASET_ID = 'gmm8_05'

    generator = GMM8DataGenerator(arrangement_radius=15, n_samples_mode=1250)
    data = generator.data
    standard = generator.standard
    plot_data(data)

    ds = Dataset(id=DATASET_ID, standard=standard, data=data)
    dir = Path(__file__).parent
    ds.save_as_dir(dir, exist_ok=True)

    cond_problem.write(dir / f'problem_{cond_problem.id}.json')
    uncond_problem.write(dir / f'problem_{uncond_problem.id}.json')

    # n_pad = 4
    # ds_padded, padding_vars = get_padded_dataset(ds, n_dimensions_padding=n_pad)
    #
    # problem_padded = ConditionalGenerativeModelingProblem(
    #     id=f'{cond_problem.id}_padded_{n_pad}',
    #     name=f'{cond_problem.name} - {n_pad} Padding Features',
    #     generate_space=Space(cond_problem.generate_space.variables + padding_vars),
    #     condition_space=cond_problem.condition_space,
    #     related_problems=[cond_problem]
    # )
    #
    # dir = Path(__file__).parent / 'wd' / ds_padded.id
    # ds_padded.save_as_dir(dir)
    #
    # problem_padded.write(dir / f'problem_{problem_padded.id}.json')
    #
    # ds_padded = Dataset.from_dir(dir)
