// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export const NVIDIA_PALETTE = [
    {
      group: "Gray Values",
      description:
        "Grays can be used to break up content in a layout or asset without competing with NVIDIA Green.",
      colors: [
        {
          name: "Gray Light 2",
          hex: "#DCDCDC",
        },
        {
          name: "Gray Light 1",
          hex: "#B4B4B4",
        },
        {
          name: "Gray",
          hex: "#999999",
        },
        {
          name: "Gray Dark 1",
          hex: "#6F6F6F",
        },
        {
          name: "Gray Dark 2",
          hex: "#444444",
        },
      ],
    },
    {
        group: "Red Values",
        colors: [
          {
            name: "Red Light 2",
            hex: "#ECD4D5",
          },
          {
            name: "Red Light 1",
            hex: "#D8A9AA",
          },
          {
            name: "Red",
            hex: "#C57F80",
          },
          {
            name: "Red Dark 1",
            hex: "#B15455",
          },
          {
            name: "Red Dark 2",
            hex: "#9E292B",
          },
        ],
    },
    {
        group: "Blue Values",
        colors: [
          {
            name: "Blue Light 2",
            hex: "#D3DCE8",
          },
          {
            name: "Blue Light 1",
            hex: "#B3C3D7",
          },
          {
            name: "Blue",
            hex: "#7D99BA",
          },
          {
            name: "Blue Dark 1",
            hex: "#5A7DA5",
          },
          {
            name: "Blue Dark 2",
            hex: "#235787",
          },
        ],
    },
    {
      group: "Green Values",
      colors: [
        {
          name: "Green Light 2",
          hex: "#D4D4B8",
        },
        {
          name: "Green Light 1",
          hex: "#B8BC96",
        },
        {
          name: "Green",
          hex: "#9AA075",
        },
        {
          name: "Green Dark 1",
          hex: "#828859",
        },
        {
          name: "Green Dark 2",
          hex: "#5E6638",
        },
      ],
    },
    {
      group: "Yellow and Orange Values",
      colors: [
        {
          name: "Yellow Light 1",
          hex: "#FEEEB2",
          rgb: [254, 238, 178],
          pms: "100 C",
        },
        {
          name: "Yellow Light 2",
          hex: "#FCDE7B",
          rgb: [252, 222, 123],
          pms: "115 C",
        },
        {
          name: "Yellow",
          hex: "#F9C500",
          rgb: [249, 197, 0],
          pms: "7548 C",
        },
        {
          name: "Orange",
          hex: "#EF9100",
          rgb: [239, 145, 0],
          pms: "3588 C",
        },
        {
          name: "Orange Dark 1",
          hex: "#DF6500",
          rgb: [223, 101, 0],
          pms: "159 C",
        },
      ],
    },
    {
        group: "Purple Values",
        colors: [
          {
            name: "Purple Light 2",
            hex: "#F9D4FF",
          },
          {
            name: "Purple Light 1",
            hex: "#C359EF",
          },
          {
            name: "Purple",
            hex: "#952FC6",
          },
          {
            name: "Purple Dark 1",
            hex: "#741D9D",
          },
          {
            name: "Purple Dark 2",
            hex: "#4D1368",
            rgb: [77, 19, 104],
            pms: "2617 C",
          },
        ],
      },
      {
      group: "Magenta Values",
      colors: [
        {
          name: "Magenta Light 2",
          hex: "#FFD3F2",
          rgb: [255, 211, 242],
          pms: "706 C",
        },
        {
          name: "Magenta Light 1",
          hex: "#FC79CA",
          rgb: [252, 121, 202],
          pms: "211 C",
        },
        {
          name: "Magenta",
          hex: "#D2308E",
          rgb: [210, 48, 142],
          pms: "2039 C",
        },
        {
          name: "Magenta Dark 1",
          hex: "#8C1C55",
          rgb: [140, 28, 85],
          pms: "675 C",
        },
        {
          name: "Magenta Dark 2",
          hex: "#5D1337",
          rgb: [93, 19, 55],
          pms: "235 C",
        },
      ],
    },
    {
      group: "Teal Values",
      colors: [
        {
          name: "Teal Light 2",
          hex: "#ADFCF8",
          rgb: [173, 252, 248],
          pms: "565 C",
        },
        {
          name: "Teal Light 1",
          hex: "#9AEFE5",
          rgb: [154, 239, 229],
          pms: "3255 C",
        },
        {
          name: "Teal",
          hex: "#1DBBA4",
          rgb: [29, 187, 164],
          pms: "3262 C",
        },
        {
          name: "Teal Dark 1",
          hex: "#0D8473",
          rgb: [13, 132, 115],
          pms: "3285 C",
        },
        {
          name: "Teal Dark 2",
          hex: "#04554B",
          rgb: [4, 85, 75],
          pms: "7729 C",
        },
      ],
    },
  ];

  export const NVIDIA_PALETTE_BY_NAME = Object.fromEntries(
    NVIDIA_PALETTE.flatMap((group) =>
      group.colors.map((color) => [color.name, { ...color, group: group.group }])
    )
  );
