import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://windily-cloud.github.io/windily-blog/",
  author: "WindilyCloud",
  desc: "Aa life-long learner and a passionate developer, I am always looking for new challenges and opportunities to learn and grow. ",
  title: "WindilyCloud",
  ogImage: "windily-og.jpg",
  lightAndDarkMode: true,
  postPerPage: 10,
};

export const LOCALE = ["zh-CN"]; // set to [] to use the environment default

export const LOGO_IMAGE = {
  enable: false,
  svg: true,
  width: 216,
  height: 46,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/windily-cloud",
    linkTitle: ` ${SITE.title} on Github`,
    active: true,
  },
  {
    name: "Mail",
    href: "mailto:www.4399li@@qq.com",
    linkTitle: `发送邮件到 ${SITE.title}`,
    active: true,
  },
  {
    name: "Discord",
    href: "https://github.com/satnaing/astro-paper",
    linkTitle: `${SITE.title} on Discord`,
    active: false,
  },
  {
    name: "Steam",
    href: "https://github.com/satnaing/astro-paper",
    linkTitle: `${SITE.title} on Steam`,
    active: false,
  },
];
