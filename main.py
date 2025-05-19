const { createClient } = require("webdav");

const client = createClient(
  "https://gift.nodisk.ir/remote.php/dav/files/09962165421",
  {
    username: "09962165421", // یا ایمیل اکانت نودیسک
    password: "156907"
  }
);

async function test() {
  const contents = await client.getDirectoryContents("/");
  console.log(contents);
}

test();
