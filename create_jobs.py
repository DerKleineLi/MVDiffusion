from pathlib import Path

scene_name = "LivingRoom-36282"
short_prompt = "classic"  # used for output_dir
prompt_stuff = "a classic style living room"
prompt_thing = "a {thing_type} in a classic style living room"
camera_xyz = [0.85963, 1.1377, -4.6706]
radius = 2.5

if __name__ == "__main__":
    config_pattern_file = "configs/scenetex_baseline_pattern.yaml"
    job_pattern_file = "jobs/pattern.job"

    with open(config_pattern_file, "r") as f:
        config_pattern = f.read()
    with open(job_pattern_file, "r") as f:
        job_pattern = f.read()

    job_dir = Path("jobs")
    config_dir = Path("configs") / "scenes"
    config_dir.mkdir(exist_ok=True, parents=True)
    mesh_file = Path("/rhome/hli/workspace/Text2Tex/data") / scene_name / "scene.obj"
    output_dir = Path("outputs") / scene_name / short_prompt

    config = config_pattern.format(
        output_dir=output_dir,
        prompt='"' + prompt_stuff + '"',
        radius=radius,
        camera_center=camera_xyz,
        mesh_file=mesh_file,
    )

    config_file = config_dir / f"{scene_name}_{short_prompt}.yaml"
    with open(config_file, "w") as f:
        f.write(config)

    job = job_pattern.replace("{config_file}", str(config_file))
    with open(job_dir / f"{scene_name}_{short_prompt}.job", "w") as f:
        f.write(job)
