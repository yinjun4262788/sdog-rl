import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco


def _to_abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _inject_mujoco_config(
    urdf_text: str,
    meshdir: str = None,
    discardvisual: bool = False,
    balanceinertia: bool = True,
    convexhull: bool = True,
    meshscale: str = None,
) -> str:
    attrs = []
    if meshdir:
        attrs.append(f'meshdir="{meshdir}"')

    attrs.append(f'discardvisual="{"true" if discardvisual else "false"}"')
    attrs.append(f'balanceinertia="{"true" if balanceinertia else "false"}"')
    attrs.append(f'convexhull="{"true" if convexhull else "false"}"')

    if meshscale:
        attrs.append(f'meshscale="{meshscale}"')

    # Ensure radians are used for angles (URDF standard)
    attrs.append('angle="radian"')

    # Add other useful defaults
    # fusestatic="false" ensures static bodies are not merged unexpectedly
    attrs.append('fusestatic="false"')

    insert = f'<mujoco><compiler {" ".join(attrs)}/></mujoco>'

    i = urdf_text.find("<robot")
    if i < 0:
        raise ValueError("Not a URDF: missing <robot ...> root element")
    gt = urdf_text.find(">", i)
    if gt < 0:
        raise ValueError("Not a URDF: malformed <robot ...> tag")
    return urdf_text[: gt + 1] + insert + urdf_text[gt + 1 :]


def post_process_mjcf(xml_path: str, rotate_root: str = None, add_freejoint: bool = True, z_offset: float = 0.5):
    print(f"Post-processing MJCF: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            print("Error: <worldbody> not found in MJCF")
            return

        bodies = list(worldbody.findall("body"))
        if not bodies:
            print("Error: No <body> found in <worldbody>")
            return

        # Handle rotate_root
        target_body = bodies[0] # Default to the first body found
        print(f"Target body for modification: {target_body.get('name')}")
        
        if rotate_root:
            print(f"Applying root rotation: {rotate_root}")
            wrapper = ET.Element("body", name="root_transform", euler=rotate_root)
            for b in bodies:
                worldbody.remove(b)
                wrapper.append(b)
            worldbody.append(wrapper)
            target_body = wrapper # Now apply freejoint/offset to the wrapper
        
        # Apply z-offset to the root body
        current_pos = target_body.get("pos", "0 0 0")
        print(f"Original pos: {current_pos}")
        try:
            vals = list(map(float, current_pos.split()))
            if len(vals) == 3:
                x, y, z = vals
                new_pos = f"{x} {y} {z + z_offset}"
                target_body.set("pos", new_pos)
                print(f"New pos with offset {z_offset}: {new_pos}")
            else:
                target_body.set("pos", f"0 0 {z_offset}")
        except ValueError:
             # If pos is malformed or missing, just set it
            target_body.set("pos", f"0 0 {z_offset}")

        # Add freejoint if requested
        if add_freejoint:
            # Check if freejoint already exists
            if target_body.find("freejoint") is None:
                 print("Adding <freejoint/>")
                 # Insert at the beginning of the body
                 target_body.insert(0, ET.Element("freejoint", name="root_joint"))
            else:
                 print("<freejoint/> already exists.")

        # Indent for prettiness (simple hack)
        if hasattr(ET, "indent"):
            ET.indent(tree, space="  ", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print("Post-processing saved successfully.")
    except Exception as e:
        print(f"Warning: Failed to post-process MJCF: {e}")


def convert_urdf_to_mjcf(
    urdf_path: str,
    out_xml_path: str,
    meshdir: str,
    discardvisual: bool = False,
    balanceinertia: bool = True,
    convexhull: bool = True,
    meshscale: str = None,
    rotate_root: str = None,
    add_freejoint: bool = True,
    z_offset: float = 0.5,
) -> None:
    urdf_path = _to_abs(urdf_path)
    out_xml_path = _to_abs(out_xml_path)

    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(urdf_path)

    os.makedirs(os.path.dirname(out_xml_path) or ".", exist_ok=True)

    urdf_dir = os.path.dirname(urdf_path)
    
    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()

    # Auto-detect meshdir if not provided
    if meshdir is None:
        # Common pattern: ../meshes relative to URDF
        candidate_meshes = os.path.join(urdf_dir, "../meshes")
        if os.path.isdir(candidate_meshes):
            # Check if URDF actually references meshes in this way
            if "../meshes/" in urdf_text or "filename=\"package://" in urdf_text:
                 print(f"Auto-detected meshdir: {os.path.abspath(candidate_meshes)}")
                 meshdir = os.path.abspath(candidate_meshes)
    
    meshdir = str(meshdir) if meshdir else None
    
    patched_text = _inject_mujoco_config(
        urdf_text, 
        meshdir=meshdir, 
        discardvisual=discardvisual,
        balanceinertia=balanceinertia,
        convexhull=convexhull,
        meshscale=meshscale
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".urdf",
            prefix="patched_",
            dir=urdf_dir,
            encoding="utf-8",
            delete=False,
        ) as tf:
            tf.write(patched_text)
            tmp_path = tf.name

        model = mujoco.MjModel.from_xml_path(tmp_path)
        mujoco.mj_saveLastXML(out_xml_path, model)
        
        # Always run post-process to handle freejoint and z-offset
        post_process_mjcf(out_xml_path, rotate_root=rotate_root, add_freejoint=add_freejoint, z_offset=z_offset)
            
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="Absolute/relative path to URDF file")
    ap.add_argument("--out", required=True, help="Output MJCF xml path")
    ap.add_argument(
        "--meshdir",
        default=None,
        help='Value for MuJoCo URDF compiler meshdir (default: None, uses URDF relative paths).',
    )
    ap.add_argument("--discardvisual", action="store_true", help="Set discardvisual=true in MuJoCo compiler")
    ap.add_argument("--no-balanceinertia", action="store_true", help="Disable balanceinertia (default: enabled)")
    ap.add_argument("--no-convexhull", action="store_true", help="Disable convexhull (default: enabled)")
    ap.add_argument("--meshscale", default=None, help="Scale factor for meshes (e.g. '0.001 0.001 0.001')")
    ap.add_argument("--rotate-root", default=None, help="Rotate root body (Euler degrees, e.g. '90 0 0')")
    ap.add_argument("--no-freejoint", action="store_true", help="Do not add freejoint to root body")
    ap.add_argument("--z-offset", type=float, default=0.5, help="Vertical offset for root body (default: 0.5m)")
    
    args = ap.parse_args()

    convert_urdf_to_mjcf(
        urdf_path=args.urdf,
        out_xml_path=args.out,
        meshdir=args.meshdir,
        discardvisual=bool(args.discardvisual),
        balanceinertia=not args.no_balanceinertia,
        convexhull=not args.no_convexhull,
        meshscale=args.meshscale,
        rotate_root=args.rotate_root,
        add_freejoint=not args.no_freejoint,
        z_offset=args.z_offset,
    )
    print("Saved MJCF:", _to_abs(args.out))


if __name__ == "__main__":
    main()